# perceptual

This is a small library for computing perceptual image hashes. Those hashes allow quickly finding duplicating images.

Currently, there's 5 different hashes implemented here:

* **DHash**. Fast "difference hash", stored as a uint64 value, meant to be compared using Hamming distance. Uses the idea from here: https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
* **HHash**. Fast, but imprecise "histogram hash". Very reliable against rotations (especially 90/180/270 degrees) and mirrorings, but can easily given false positives or even false negatives (if colors were changed too much).
* **PHash**. DCT hash, used in phash.org library and described in http://phash.org/docs/pubs/thesis_zauner.pdf
* **MHash**. Marr hash, used in phash.org library and described in http://phash.org/docs/pubs/thesis_zauner.pdf. Current implementation is very slow and not very reliable; not recommended to use in the current state.
* **RHash**. Radial hash, used in phash.org library and described in http://phash.org/docs/pubs/thesis_zauner.pdf. Very slow (relative to other hashes) when comparing hashes.

In the future SIFT/SURF feature extraction can be added to the library. This should allow the most precise image matching. However it will also be the slowest one.

[API Reference](https://denull.github.io/perceptual/)

## Installation

```
nimble install perceptual
```

## Example

```nim
import perceptual

let hash1 = dhash("image1.jpg")
let hash2 = dhash("image2.jpg")

let difference = diff(hash1, hash2)

echo "Image Difference: ", difference
if difference < 15:
  echo "Probably the same image!"
elif difference < 25:
  echo "Can be similar"
else:
  echo "Different"
```