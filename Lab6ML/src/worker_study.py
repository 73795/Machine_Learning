import time
from data import get_data


def main():
    workers = [0, 1, 2, 4, 8]
    for w in workers:
        start = time.time()
        train_loader, _, _ = get_data(
            batch_size=128,
            num_workers=w
        )
        for images, labels in train_loader:
            pass
        elapsed = time.time() - start
        print(f"Workers: {w}, "f"Time: {elapsed:.2f}")
if __name__ == "__main__":
    main()