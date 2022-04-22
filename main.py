
''' 1. 依赖导入 '''

# ------------------------------------------------------------------------------------------------------------
import tqdm
import random
import numpy as np

import megengine as mge
import megengine.module as M
import megengine.functional as F
from megengine.utils.module_stats import module_stats
# ------------------------------------------------------------------------------------------------------------

''' 2. 参数设置 '''

# ------------------------------------------------------------------------------------------------------------
mge.set_default_device('gpu0')

seed = 42
batchsz = 8
patchsz = 224
num_valid = 800
max_epochs = 256
learning_rate = 1e-3

np.random.seed(seed)
mge.random.seed(seed)
rnd = random.Random(seed)

train_data_path = 'dataset/burst_raw/competition_train_input.0.2.bin'  # 训练集样本保存路径
train_target_path = 'dataset/burst_raw/competition_train_gt.0.2.bin'  # 训练集标签保存路径 
test_data_path = 'dataset/burst_raw/competition_test_input.0.2.bin'  # 测试集样本保存路径

model_save_path = 'workspace/model'  # 模型权重保存路径
model_newest_save_path = 'workspace/model_newest'
result_save_path = 'workspace/submit/result.bin'  # 测试集预测结果保存路径
train_pred_path = 'workspace/train_pred.bin'
train_pred_path0 = 'workspace/train_pred0.bin'
# ------------------------------------------------------------------------------------------------------------

''' 3. 模型定义 '''

# ------------------------------------------------------------------------------------------------------------
# baseline: https://github.com/megvii-research/PMRID

def DepthwiseConv(in_channels, kernel_size, stride, padding):
    return M.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, groups=in_channels, bias=False)


def PointwiseConv(in_channels, out_channels):
    return M.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=True)


class CovSepBlock(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0): 
        super().__init__()
        self.dc = DepthwiseConv(in_channels, kernel_size, stride=stride, padding=padding)
        self.pc = PointwiseConv(in_channels, out_channels)
        self.dc2 = DepthwiseConv(out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.dc(x)
        x1 = self.pc(x)
        x = self.dc2(x1) + x1
        return x


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return M.Sequential(DepthwiseConv(in_channels, kernel_size, stride=stride, padding=(kernel_size//2)), 
                         PointwiseConv(in_channels, out_channels)
    )


class CALayer(M.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = M.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = M.Sequential(
                M.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                M.ReLU(),
                M.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                M.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(M.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=4, bias=False):
        super(CAB, self).__init__()

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = M.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), 
                                  M.LeakyReLU(0.125),
                                  conv(n_feat, n_feat, kernel_size, bias=bias),
        )

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class Encoder(M.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sepconv = CovSepBlock(in_channels, out_channels * 2, padding=2)
        self.activate = M.LeakyReLU(0.125)  
        self.sepconv2 = CovSepBlock(out_channels * 2, out_channels, padding=2)
        self.proj = M.Identity()
        if in_channels != out_channels:
            self.proj = CovSepBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.cab = CAB(out_channels)

    def forward(self, x):
        branch = self.proj(x)
        x = self.sepconv(x)
        x = self.activate(x)
        x = self.sepconv2(x)
        x += branch
        x = self.cab(x)
        return x


class Upsampling(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()
        self.upsample = M.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2)

    def forward(self, x):
        return self.upsample(x)


class Downsampling(M.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sepconv = CovSepBlock(in_channels=in_channels, out_channels=out_channels * 2, stride=2, padding=2)
        self.activate = M.LeakyReLU(0.125) 
        self.sepconv2 = CovSepBlock(in_channels=out_channels * 2, out_channels=out_channels, padding=2)
        self.branchconv = CovSepBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.cab = CAB(out_channels)

    def forward(self, x):
        branch = x
        x = self.sepconv(x)
        x = self.activate(x)
        x = self.sepconv2(x)
        branch = self.branchconv(branch)
        x += branch
        x = self.cab(x) 
        return x 


class Decoder(M.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sepconv = CovSepBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.activate = M.LeakyReLU(0.125)  
        self.sepconv2 = CovSepBlock(out_channels, out_channels, kernel_size=3, padding=1)
        self.cab = CAB(out_channels)

    def forward(self, x):
        branch = x
        x = self.sepconv(x)
        x = self.activate(x)
        x = self.sepconv2(x)
        x = x + branch
        x = self.cab(x)
        return x


def EncoderStage(in_channels, out_channels, num_encoder):
    seq = [
        Downsampling(in_channels, out_channels),
    ]
    for _ in range(num_encoder):
        seq.append(
            Encoder(out_channels, out_channels)
        )
    return M.Sequential(*seq)


class DecoderStage(M.Module):
    def __init__(self, in_channels, out_channels, skip_in_channels):
        super().__init__()
        self.decoder = Decoder(in_channels, in_channels)
        self.upsampling = Upsampling(in_channels, out_channels)
        self.skipconnect = CovSepBlock(skip_in_channels, out_channels, kernel_size=3, padding=1)
        self.activate = M.LeakyReLU(0.125) 
        self.cab = CAB(out_channels)


    def forward(self, x):
        input, skip = x
        input = self.decoder(input)
        input = self.upsampling(input)
        skip = self.skipconnect(skip)
        skip = self.activate(skip)
        output = input + skip
        output = self.cab(output)
        return output


class Net(M.Module):
    def __init__(self):
        super().__init__()
        self.conv = M.Conv2d(in_channels=1, out_channels=38, kernel_size=5, padding=2)
        self.relu = M.LeakyReLU(0.125)
        self.encoder_stage1 = EncoderStage(in_channels=38, out_channels=32, num_encoder=1)
        self.encoder_stage2 = EncoderStage(in_channels=32, out_channels=26, num_encoder=1)

        self.encoder_stage3 = EncoderStage(in_channels=26, out_channels=18, num_encoder=1)
        self.enc2dec = CovSepBlock(in_channels=18, out_channels=18, kernel_size=3, padding=1)
        self.med_activate = M.LeakyReLU(0.125)

        self.decoder_stage2 = DecoderStage(in_channels=18, skip_in_channels=26, out_channels=26)
        self.decoder_stage3 = DecoderStage(in_channels=26, skip_in_channels=32, out_channels=32)
        self.decoder_stage4 = DecoderStage(in_channels=32, skip_in_channels=38, out_channels=38)
        self.output_layer = M.Sequential(*(Decoder(in_channels=38, out_channels=38),
                                           M.Conv2d(in_channels=38, out_channels=1, kernel_size=3, padding=1)))
        
    def forward(self, img):
        assert img.shape[1] == 1

        pre = self.conv(img)
        pre = self.relu(pre)
        first = self.encoder_stage1(pre)
        second = self.encoder_stage2(first)
        
        med = self.encoder_stage3(second)
        med = self.enc2dec(med)
        med = self.med_activate(med)

        de_second = self.decoder_stage2((med, second))
        de_thrid = self.decoder_stage3((de_second, first))
        de_fourth = self.decoder_stage4((de_thrid, pre))

        output = self.output_layer(de_fourth)
        output = output + img
        return output
# ------------------------------------------------------------------------------------------------------------

''' 4. 参数分析 '''

# ------------------------------------------------------------------------------------------------------------
net = Net()

input_data = np.random.rand(1, 1, 256, 256).astype("float32")

total_stats, stats_details = module_stats(
    net,
    inputs=(input_data,),
    cal_params=True,
    cal_flops=True,
    logging_to_stdout=False, # True,
)

print("params %.3fK MAC/pixel %.0f"%(total_stats.param_dims/1e3, total_stats.flops/input_data.shape[2]/input_data.shape[3]))
# ------------------------------------------------------------------------------------------------------------

''' 5. 数据加载与增强 '''

# ------------------------------------------------------------------------------------------------------------
print('loading data ...')

content_ref = open(train_data_path, 'rb').read()
samples_ref = np.frombuffer(content_ref, dtype='uint16').reshape((-1, 256, 256))

content_target = open(train_target_path, 'rb').read()
samples_gt = np.frombuffer(content_target, dtype='uint16').reshape((-1, 256, 256))

print(samples_ref.shape, samples_gt.shape)


samples_ref_new = [samples_ref[:, 0:patchsz, 0:patchsz], samples_ref[:, 0:patchsz, 256-patchsz:256],
                   samples_ref[:, 256-patchsz:256, 0:patchsz], samples_ref[:, 256-patchsz:256, 256-patchsz:256]
]

samples_gt_new = [samples_gt[:, 0:patchsz, 0:patchsz], samples_gt[:, 0:patchsz, 256-patchsz:256],
                  samples_gt[:, 256-patchsz:256, 0:patchsz], samples_gt[:, 256-patchsz:256, 256-patchsz:256]
]

aug_modes = {0: [False, False, False], 1: [False, False, True], 
             2: [False, True, False], 3: [False, True, True],
             4: [True, False, False],  5: [True, False, True],  
             6: [True, True, False],  7: [True, True, True]
}

def argument(raw: np.ndarray, flip_h: bool, flip_w: bool, transpose: bool) -> np.ndarray:
    out = raw
    if flip_h:
        out = out[::-1, :]
    if flip_w:
        out = out[:, ::-1]
    if transpose:
        out = out.T
    return out
# ------------------------------------------------------------------------------------------------------------

''' 6. 模型训练 '''

# ------------------------------------------------------------------------------------------------------------
'''
opt = mge.optimizer.Adam(net.parameters(), lr=learning_rate)
gm = mge.autodiff.GradManager().attach(net.parameters())

print('training...')


losses = []
best_score = -1.0
num_iters = len(samples_gt) * len(samples_gt_new) // batchsz

for epoch in range(max_epochs):

    order = np.random.choice(len(samples_gt_new), len(samples_gt_new), replace=False)

    samples_ref = np.concatenate((samples_ref_new[order[0]], 
                                 samples_ref_new[order[1]], 
                                 samples_ref_new[order[2]], 
                                 samples_ref_new[order[3]]), 0)

    samples_gt = np.concatenate((samples_gt_new[order[0]], 
                                samples_gt_new[order[1]], 
                                samples_gt_new[order[2]], 
                                samples_gt_new[order[3]]), 0)

    
    for g in opt.param_groups:
        g['lr'] = learning_rate * (max_epochs - epoch) / max_epochs
    opt.zero_grad()

    for it in range(num_iters):
        batch_inp_np = np.zeros((batchsz, 1, patchsz, patchsz), dtype='float32')
        batch_out_np = np.zeros((batchsz, 1, patchsz, patchsz), dtype='float32')
        idx = 0
        for i in range(it*batchsz, (it+1)*batchsz):
            sample = np.float32(samples_ref[i, :, :]) * np.float32(1 / 65536)
            target = np.float32(samples_gt[i, :, :]) * np.float32(1 / 65536)

            # 几何变换
            mode = aug_modes[np.random.randint(0, 8)]
            sample = argument(raw=sample, flip_h=mode[0], flip_w=mode[1], transpose=mode[2])
            target = argument(raw=target, flip_h=mode[0], flip_w=mode[1], transpose=mode[2])

            batch_inp_np[idx, 0, :, :] = sample
            batch_out_np[idx, 0, :, :] = target
            idx += 1

        batch_inp = mge.tensor(batch_inp_np)
        batch_out = mge.tensor(batch_out_np)

        with gm:
            pred = net(batch_inp)
            loss = F.nn.l1_loss(pred, batch_out, reduction='mean')
            gm.backward(loss)
            opt.step().clear_grad()

        loss = float(loss.numpy())
        losses.append(loss)

        if it % 1000 == 0:
            print(f"epoch: {epoch}, iter: {it} ...")

    fout = open(model_newest_save_path, 'wb')
    mge.save(net.state_dict(), fout)
    fout.close()

    ### valid ###
    if epoch % eval_interval == 0:
        net.eval()

        content_valid = open(train_data_path, 'rb').read()
        samples_ref_valid = np.frombuffer(content_valid, dtype='uint16').reshape((-1, 256, 256))[:num_valid, :, :]

        print('validing set predicting ...')
        fout = open(train_pred_path, 'wb')
        for i in tqdm.tqdm(range(0, len(samples_ref_valid), batchsz)):
            i_end = min(i+batchsz, len(samples_ref_valid))
            batch_inp = mge.tensor(np.float32(samples_ref_valid[i:i_end, None, :, :]) * np.float32(1/65536))
            pred = net(batch_inp)
            pred = (pred.numpy()[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')
            fout.write(pred.tobytes())
        fout.close()

#         print('loading validing set predictions ...')
        content_pred_valid = open(train_pred_path, 'rb').read()
        samples_pred_valid = np.float32(np.frombuffer(content_pred_valid, dtype='uint16').reshape((-1,256,256)))
        content_gt_valid = open(train_target_path, 'rb').read()
        samples_gt_valid = np.float32(np.frombuffer(content_gt_valid, dtype='uint16').reshape((-1,256,256)))[:num_valid, :, :]

#         print(f"computing ...")
        means = samples_gt_valid.mean(axis=(1, 2))
        weight = (1 / means) ** 0.5
        diff = np.abs(samples_pred_valid - samples_gt_valid).mean(axis=(1, 2))
        diff = diff * weight
        score = diff.mean()
        score = np.log10(100 / score) * 5
        print(f'[{epoch}/{max_epochs}] curr score: {score}, train loss: {loss}')

        if score > best_score:
            best_score = score
            print(f"[{epoch}/{max_epochs}] best score {best_score}, model saved !!! ")
            fout = open(model_save_path, 'wb')
            mge.save(net.state_dict(), fout)
            fout.close()

        net.train()
'''
# --------------------------------------------------------------------------------------------------------------

''' 7. 推理测试 '''

# ------------------------------------------------------------------------------------------------------------
def argument_reverse(raw: np.ndarray, flip_h: bool, flip_w: bool, transpose: bool) -> np.ndarray:
    out = raw
    if transpose:
        out = out.T
    if flip_w:
        out = out[:, ::-1]
    if flip_h:
        out = out[::-1, :]

    return out


aug_modes = {0: [False, False, False], 1: [False, False, True], 2: [False, True, False], 3: [False, True, True],
             4: [True, False, False],  5: [True, False, True],  6: [True, True, False],  7: [True, True, True]
}

content = open(test_data_path, 'rb').read()
samples_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))

fout = open(result_save_path, 'wb')

# loading model
model = Net()
model.load_state_dict(mge.load(model_save_path))  
model.eval()

print('testing set 1024 predicting ...')
for i in tqdm.tqdm(range(len(samples_ref))):
    # 输入 + 增强
    batch_inp_np = np.zeros((8, 1, 256, 256), dtype='float32')
    sample = np.float32(samples_ref[i, :, :]) * np.float32(1/65536)
    for k in range(len(aug_modes)):
        mode = aug_modes[k]
        batch_inp_np[k, 0, :, :] = argument(raw=sample, flip_h=mode[0], flip_w=mode[1], transpose=mode[2])
    batch_inp = mge.tensor(batch_inp_np)

    # 预测
    pred = model(batch_inp).numpy()

    # 输出 + 还原
    batch_pred_np = np.zeros((1, 256, 256), dtype='float32')
    for k in range(len(aug_modes)):
        mode = aug_modes[k]
        batch_pred_np[0, :, :] += argument_reverse(raw=pred[k, 0, :, :], flip_h=mode[0], flip_w=mode[1], transpose=mode[2]) / 8
    batch_pred_np = (batch_pred_np * 65536).clip(0, 65535).astype('uint16')

    fout.write(batch_pred_np.tobytes())
fout.close()
# ------------------------------------------------------------------------------------------------------------

''' 8. 离线验证 '''

# ------------------------------------------------------------------------------------------------------------
'''
def argument_reverse(raw: np.ndarray, flip_h: bool, flip_w: bool, transpose: bool) -> np.ndarray:
    out = raw
    if transpose:
        out = out.T
    if flip_w:
        out = out[:, ::-1]
    if flip_h:
        out = out[::-1, :]
    return out

aug_modes = {0: [False, False, False], 1: [False, False, True], 2: [False, True, False], 3: [False, True, True],
             4: [True, False, False],  5: [True, False, True],  6: [True, True, False],  7: [True, True, True]
}


content = open(train_data_path, 'rb').read()
samples_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))[:800, :, :]


fout = open(train_pred_path0, 'wb')

# loading model
model = Net()
model.load_state_dict(mge.load(model_save_path))
model.eval()

print('training set 800 predicting ...')
for i in tqdm.tqdm(range(len(samples_ref))):
    # 输入 + 增强
    batch_inp_np = np.zeros((8, 1, 256, 256), dtype='float32')
    sample = np.float32(samples_ref[i, :, :]) * np.float32(1/65536)
    for k in range(len(aug_modes)):
        mode = aug_modes[k]
        batch_inp_np[k, 0, :, :] = argument(raw=sample, flip_h=mode[0], flip_w=mode[1], transpose=mode[2])
    batch_inp = mge.tensor(batch_inp_np)

    # 预测
    pred = model(batch_inp).numpy()

    # 输出 + 还原
    batch_pred_np = np.zeros((1, 256, 256), dtype='float32')
    for k in range(len(aug_modes)):
        mode = aug_modes[k]
        batch_pred_np[0, :, :] += argument_reverse(raw=pred[k, 0, :, :], flip_h=mode[0], flip_w=mode[1], transpose=mode[2]) / 8
    batch_pred_np = (batch_pred_np * 65536).clip(0, 65535).astype('uint16')

    fout.write(batch_pred_np.tobytes())
fout.close()


# 计算 score

print('loading training set predictions ...')
content = open(train_pred_path0, 'rb').read()
samples_pred = np.float32(np.frombuffer(content, dtype='uint16').reshape((-1,256,256)))
content = open(train_target_path, 'rb').read()
samples_gt = np.float32(np.frombuffer(content, dtype='uint16').reshape((-1,256,256)))[:800, :, :]

print(f"computing ...")
means = samples_gt.mean(axis=(1, 2))
weight = (1 / means) ** 0.5
diff = np.abs(samples_pred - samples_gt).mean(axis=(1, 2))
diff = diff * weight
score = diff.mean()
score = np.log10(100 / score) * 5

print(f'training set predicting score: {score}')
'''
# ------------------------------------------------------------------------------------------------------------
