import pooch


if __name__ == "__main__":
    dataset_url = "https://zenodo.org/records/10925855/files/noisy.tiff?download=1"

    file = pooch.retrieve(
        url=dataset_url,
        known_hash="ff12ee5566f443d58976757c037ecef8bf53a00794fa822fe7bcd0dd776a9c0f",
        path="./test/input/images/image-stack-structured-noise/",
    )