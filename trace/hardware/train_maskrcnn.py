"""Train the hardware segmenter from locally annotated RGB images and instance masks.

Manifest: a JSON list of {image, mask, labels} records. Paths are relative to the
manifest; mask is a single-channel PNG with 0=background, positive instance IDs.
labels maps each instance ID (as a string) to a class in CLASS_ID_TO_NAME.
No images, masks, or recordings are included in the release.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--initialize', type=Path, help='compatible local state_dict for fine-tuning')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=.005)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        ap.error('epochs, batch size, and learning rate must be positive')
    if args.output.exists():
        ap.error('output exists; choose a new checkpoint path')
    import numpy as np
    from PIL import Image
    import torch
    from torchvision.transforms.functional import to_tensor
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sim'))
    from isaacgymenvs.open_loop.perceive_scene import build_maskrcnn, CLASS_ID_TO_NAME
    torch.manual_seed(args.seed)
    rows = json.loads(args.manifest.read_text())
    if not rows:
        ap.error('empty training manifest')

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            image = to_tensor(Image.open(args.manifest.parent / row['image']).convert('RGB'))
            instance = np.array(Image.open(args.manifest.parent / row['mask']))
            if instance.ndim != 2 or tuple(instance.shape) != tuple(image.shape[-2:]):
                raise ValueError('Mask must be single-channel and match image size')
            boxes, masks, labels = [], [], []
            for identity in np.unique(instance):
                if identity == 0:
                    continue
                label = int(row['labels'][str(int(identity))])
                if label not in CLASS_ID_TO_NAME:
                    raise ValueError('Unknown class: ' + str(label))
                mask = instance == identity
                y, x = np.nonzero(mask)
                boxes.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
                masks.append(mask)
                labels.append(label)
            if not boxes:
                raise ValueError('Each training image needs at least one annotated object')
            return image, dict(boxes=torch.tensor(boxes, dtype=torch.float32),
                               labels=torch.tensor(labels, dtype=torch.int64),
                               masks=torch.tensor(np.asarray(masks), dtype=torch.uint8))

    loader = torch.utils.data.DataLoader(Dataset(), batch_size=args.batch_size,
                                        shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))
    model = build_maskrcnn()
    if args.initialize:
        model.load_state_dict(torch.load(args.initialize, map_location='cpu'))
    model.to(args.device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=.0005)
    for epoch in range(args.epochs):
        total = 0.
        for images, targets in loader:
            losses = model([image.to(args.device) for image in images],
                           [{k: v.to(args.device) for k, v in target.items()} for target in targets])
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f'epoch {epoch + 1}: loss {total / len(loader):.6f}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), args.output)


if __name__ == '__main__':
    main()
