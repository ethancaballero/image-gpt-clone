import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import os

from module import ImageGPT
from prepare_data import unquantize

#from moviepy.editor import ImageSequenceClip


def generate(args, model, context, length, num_samples=1, temperature=1.0):

    output = context.unsqueeze(-1).repeat_interleave(
        num_samples, dim=-1
    )  # add batch so shape [seq len, batch]

    #import pdb; pdb.set_trace()

    """
    cond_logprobs = torch.zeros(0, num_samples, dtype=torch.float).unsqueeze(-1).repeat_interleave(
        num_samples, dim=-1
    ) 
    #"""

    cond_logprobs = torch.zeros(0, num_samples, 16)

    lls = torch.zeros(args.bs)

    frames = []
    pad = torch.zeros(1, num_samples, dtype=torch.long)  # to pad prev output
    if torch.cuda.is_available():
        pad = pad.cuda()
    with torch.no_grad():
        for _ in tqdm(range(length), leave=False):
            #import pdb; pdb.set_trace()
            logits = model(torch.cat((output, pad), dim=0))
            logits = logits[-1, :, :] / temperature
            logprobs = F.log_softmax(logits, dim=-1)
            #import pdb; pdb.set_trace()
            cond_logprobs = torch.cat((cond_logprobs, logprobs.unsqueeze(0)), dim=0)
            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs, num_samples=1).transpose(1, 0)
            lls += torch.gather(logprobs, -1, pred.squeeze(0).unsqueeze(-1)).squeeze(1)
            #import pdb; pdb.set_trace()
            output = torch.cat((output, pred), dim=0)
            #frames.append(output.cpu().numpy().transpose())
    return cond_logprobs.permute(1,0,2), output.permute(1,0).long(), lls


def main(args):
    model = ImageGPT.load_from_checkpoint(args.checkpoint).gpt
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    context = torch.zeros(0, dtype=torch.long)
    if torch.cuda.is_available():
        context = context.cuda()

    lps_all = torch.zeros(0, 784, 16)
    imgs_all = torch.zeros(0, 784, dtype=torch.long)
    lls_all = torch.zeros(0)

    datapoints = args.datapoints

    with torch.no_grad():
        for iters in range(datapoints//args.bs):
            lps, imgs, lls = generate(args, model, context, 28 * 28, num_samples=args.bs)

            lps_all = torch.cat((lps_all, lps), dim=0)
            imgs_all = torch.cat((imgs_all, imgs), dim=0)
            lls_all = torch.cat((lls_all, lls), dim=0)

    if not os.path.exists('_data'):
        os.makedirs('_data')
    torch.save(lps_all, '_data/cond_ll.pt')
    torch.save(imgs_all, '_data/imgs.pt')
    torch.save(lls_all, '_data/lls.pt')

    """
    pad_frames = []
    for frame in frames:
        pad = ((0, 0), (0, 28 * 28 - frame.shape[1]))
        pad_frames.append(np.pad(frame, pad_width=pad))

    pad_frames = np.stack(pad_frames)
    f, n, _ = pad_frames.shape
    pad_frames = pad_frames.reshape(f, args.rows, args.cols, 28, 28)
    pad_frames = pad_frames.swapaxes(2, 3).reshape(f, 28 * args.rows, 28 * args.cols)
    pad_frames = pad_frames[..., np.newaxis] * np.ones(3) * 17
    pad_frames = pad_frames.astype(np.uint8)

    clip = ImageSequenceClip(list(pad_frames)[:: args.downsample], fps=args.fps).resize(
        args.scale
    )
    clip.write_gif("out.gif", fps=args.fps)
    #"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--downsample", type=int, default=5)
    parser.add_argument("--datapoints", type=int, default=262144)
    args = parser.parse_args()
    main(args)
