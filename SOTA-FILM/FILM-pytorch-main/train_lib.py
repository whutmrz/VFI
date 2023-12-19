from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
import torch

from utils import to_gpu, save_checkpoint, load_checkpoint, metrics, log_image, InputPadder

def train(args, model, summary, optimizer, create_losses_fn, dataloader_train, dataloader_test, eval_loop_fn=None, eval_datasets=None, resume=None):

    loss_functions = create_losses_fn(['l1', 'perceptual'])
    PSNR = PeakSignalNoiseRatio().cuda()
    SSIM = StructuralSimilarityIndexMeasure().cuda()

    best_psnr = 0
    best_psnr_epoch = 0
    best_ssim = 0
    best_ssim_epoch = 0
    global_step = 0
    for epoch in range(args.epoch):

        # metric_psnr, metric_ssim = 0, 0
        print('training')
        model.train()
        for i, batch in enumerate(tqdm(dataloader_train)):
            batch = to_gpu(batch)
            padder = InputPadder(batch['x0'].shape, divisor=32)
            batch['x0'] = padder.pad(batch['x0'])
            batch['x1'] = padder.pad(batch['x1'])
            batch['y'] = padder.pad(batch['y'])
            model.zero_grad()
            predictions = model(to_gpu(batch))
            optimizer.zero_grad()
            losses = []
            for (loss_function) in loss_functions:
                loss = loss_functions[loss_function](batch, predictions)
                losses.append(loss)
            loss = sum(losses)
            loss.backward()
            optimizer.step()
            global_step+=1

            summary.add_scalar('train/loss', float(loss), global_step=global_step)
        save_checkpoint(args, model, optimizer, epoch)
        model.eval()
        metric_psnr, metric_ssim = 0, 0
        for i, batch in enumerate(tqdm(dataloader_test)):
            test_step = 0
            batch = to_gpu(batch)
            padder = InputPadder(batch['x0'].shape, divisor=32)
            batch['x0'] = padder.pad(batch['x0'])
            batch['x1'] = padder.pad(batch['x1'])
            batch['y'] = padder.pad(batch['y'])
            with torch.no_grad():
                predictions = model(to_gpu(batch))
                test_step+=1
            batch['x0'] = padder.unpad(batch['x0'])
            batch['x1'] = padder.unpad(batch['x1'])
            batch['y'] = padder.unpad(batch['y'])
            predictions['image'] = padder.unpad(predictions['image'])
            psnr, ssim = metrics(predictions, batch, summary, PSNR, SSIM, test_step)
            metric_psnr += psnr
            metric_ssim += ssim
            if i == len(dataloader_test)-1:
                log_image(batch, predictions, args, summary, epoch, i, test_step)
                # with torch.no_grad():
                #     psnr, ssim = metrics(predictions, batch, summary, PSNR, SSIM, global_step)
                #     metric_psnr += psnr
                #     metric_ssim += ssim
        if float(metric_psnr)/len(dataloader_test) > best_psnr:
            best_psnr = float(metric_psnr)/len(dataloader_test)
            best_psnr_epoch = epoch+1
        if float(metric_ssim)/len(dataloader_test) > best_ssim:
            best_ssim = float(metric_ssim)/len(dataloader_test)
            best_ssim_epoch = epoch+1
        print('epoch=',epoch+1, 'psnr=', float(metric_psnr)/len(dataloader_test), 'best_psnr=', best_psnr, 'best_psnr_epoch=', best_psnr_epoch)
        print('epoch=',epoch+1, 'ssim=', float(metric_ssim)/len(dataloader_test), 'best_ssim=', best_ssim, 'best_ssim_epoch=', best_ssim_epoch)
        # summary.add_scalar('train/psnr_epoch', float(metric_psnr/(len(dataloader))*100), global_step=epoch)
        # summary.add_scalar('train/ssim_epoch', float(metric_ssim/(len(dataloader))*100), global_step=epoch)

       
