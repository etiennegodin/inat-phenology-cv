from pytorch_pipeline.train.batch_sampler import MaxImagesBatchSampler


def test_max_images_batch_sampler_len():
    # Bag sizes for 10 observations
    bag_sizes = [2, 3, 1, 5, 2, 8, 4, 1, 3, 2]
    max_images = 6

    sampler = MaxImagesBatchSampler(
        bag_sizes=bag_sizes, max_images=max_images, shuffle=False
    )
    batches = list(sampler)

    # Verify len(sampler) matches actual batch count yielded
    assert len(sampler) == len(batches)
    # Batches (5) should be less than observations (10)
    assert len(sampler) < len(bag_sizes)


def test_max_images_batch_sampler_shuffled():
    bag_sizes = [1, 2, 3, 4, 5, 2, 1, 3]
    max_images = 5

    sampler = MaxImagesBatchSampler(
        bag_sizes=bag_sizes, max_images=max_images, shuffle=True
    )
    batches = list(sampler)

    # Length estimate should closely match yielded batch count
    assert abs(len(sampler) - len(batches)) <= 1
