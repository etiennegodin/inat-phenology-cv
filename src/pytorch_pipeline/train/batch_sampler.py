import random

from torch.utils.data import Sampler


class MaxImagesBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        bag_sizes: list[int],
        max_images: int,
        shuffle: bool = True,
    ):
        self.bag_sizes = bag_sizes
        self.max_images = max_images
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.bag_sizes)))

        if self.shuffle:
            random.shuffle(indices)

        batch = []
        total_images = 0

        for idx in indices:
            bag_size = self.bag_sizes[idx]

            # Large bags become their own batch
            if bag_size >= self.max_images:
                if batch:
                    yield batch
                    batch = []
                    total_images = 0

                yield [idx]
                continue

            if total_images + bag_size > self.max_images:
                yield batch
                batch = []
                total_images = 0

            batch.append(idx)
            total_images += bag_size

        if batch:
            yield batch

    def __len__(self):
        # approximate
        return len(self.bag_sizes)
