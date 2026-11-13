"""Download selected cases from the official MSD tar using S3 byte ranges."""
import argparse
import random
import tarfile
from pathlib import Path, PurePosixPath

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ConnectionClosedError, EndpointConnectionError, ReadTimeoutError


class S3RangeReader:
    """Small seekable file wrapper that fetches cached byte ranges from S3."""
    def __init__(self, client, bucket, key, size, block_size=256 * 1024):
        self.client, self.bucket, self.key = client, bucket, key
        self.size, self.block_size, self.position = size, block_size, 0
        self.cache_offset, self.cache = -1, b""

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.position + offset
        elif whence == 2:
            position = self.size + offset
        else:
            raise ValueError("Invalid whence")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = min(position, self.size)
        return self.position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        output = bytearray()
        while size:
            block_offset = (self.position // self.block_size) * self.block_size
            if block_offset != self.cache_offset:
                network_errors = (ConnectionClosedError, EndpointConnectionError, ReadTimeoutError)
                for attempt in range(4):
                    end = min(block_offset + self.block_size, self.size) - 1
                    response = self.client.get_object(
                        Bucket=self.bucket, Key=self.key,
                        Range=f"bytes={block_offset}-{end}",
                    )
                    try:
                        self.cache = response["Body"].read()
                        break
                    except network_errors:
                        response["Body"].close()
                        if attempt == 3:
                            raise
                        self.block_size = max(16 * 1024, self.block_size // 2)
                    finally:
                        response["Body"].close()
                self.cache_offset = block_offset
            within = self.position - self.cache_offset
            amount = min(size, len(self.cache) - within)
            if amount <= 0:
                break
            output.extend(self.cache[within:within + amount])
            self.position += amount
            size -= amount
        return bytes(output)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a small labeled MSD BrainTumour subset without downloading the full 7.6 GB archive"
    )
    parser.add_argument("--data", default="data/Task01_BrainTumour")
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cases < 3:
        parser.error("--cases must be at least 3 for train/validation/test splitting")

    bucket = "msd-for-monai"
    key = "Task01_BrainTumour.tar"
    client = boto3.client(
        "s3", region_name="us-west-2",
        config=Config(
            signature_version=UNSIGNED,
            connect_timeout=20,
            read_timeout=120,
            retries={"max_attempts": 8, "mode": "standard"},
        ),
    )
    try:
        size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except Exception as exc:
        raise RuntimeError(f"Could not access public archive s3://{bucket}/{key}: {exc}") from exc

    print("Scanning the remote archive index (this can take a few minutes)...", flush=True)
    remote_file = S3RangeReader(client, bucket, key, size)
    with tarfile.open(fileobj=remote_file, mode="r:") as archive:
        members = archive.getmembers()
        images, labels = {}, {}
        for member in members:
            if not member.isfile() or not member.name.endswith(".nii.gz"):
                continue
            path = PurePosixPath(member.name)
            if "imagesTr" in path.parts:
                images[path.name] = member
            elif "labelsTr" in path.parts:
                labels[path.name] = member
        pairs = [(name, image, labels[name]) for name, image in images.items() if name in labels]
        pairs.sort(key=lambda item: item[0])
        if not pairs:
            raise RuntimeError(
                "The official archive was reachable, but no matching imagesTr/labelsTr NIfTI pairs "
                "were found. The archive layout may have changed."
            )
        if args.cases > len(pairs):
            raise ValueError(f"Requested {args.cases} cases but found {len(pairs)} labeled cases")

        random.Random(args.seed).shuffle(pairs)
        selected = sorted(pairs[:args.cases], key=lambda item: item[0])
        root = Path(args.data)
        for folder in ("imagesTr", "labelsTr"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        total_bytes = 0
        for name, image_member, label_member in selected:
            for folder, member in (("imagesTr", image_member), ("labelsTr", label_member)):
                content = archive.extractfile(member).read()
                (root / folder / name).write_bytes(content)
                total_bytes += len(content)
            print(name.removesuffix(".nii.gz"))

        # Keep task metadata and license if present in the official archive.
        for member in members:
            base = PurePosixPath(member.name).name
            if base in {"dataset.json", "LICENSE.txt", "license.txt"} and member.isfile():
                (root / base).write_bytes(archive.extractfile(member).read())

    print(f"Downloaded {len(selected)} paired cases ({total_bytes / (1024**2):.1f} MiB) to {root}")
    print(f"Subset selection seed: {args.seed}")


if __name__ == "__main__":
    main()