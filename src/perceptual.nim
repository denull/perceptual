import bitops, options, strformat, algorithm
import pixie, chroma

# Five types of perceptive hashes implemented:
# - dhash (difference hash), based on https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
#   Returns uint64 value; can be compared using Hamming distance (popcount). Distance therefore is between 0 and 64.
# - hhash (histogram hash)
#   Returns uint64 value; needs special comparison (because it consists of 16 four-bit values). Distance is between 0 and 240 (16x15).
# - phash (DCT hash), based on http://phash.org/docs/pubs/thesis_zauner.pdf
#   Returns uint64 value; can be compared using Hamming distance (popcount). Distance therefore is between 0 and 64. Threshold value of 22 is recommended by pHash.
# - mhash (Marr hash), used by pHash library. Distance is between 0 and 576.
# - rhash (radial hash), used by pHash library. Distance is a floating-point value between 0.0 and 100.0
# TODO:
# - shash (SURF hash)


type
  DHash* = uint64 ##\
  ## DHash is a 64-bit hash, so it's just a synonym for `uint64`.
  ## Based on the idea described at https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
  ## This hash is fast to compute and compare (using Hamming distance), and it produces decent results.
  
  HHash* = distinct uint64 ##\
  ## HHash is a 64-bit hash, so it's just a synonym for `uint64`. However, it should not be compared using `popcount`, so it's marked as a distinct type.
  ## Basic histogram hash. Can produce a lot of false positives, but it's the most reliable against rotations.
  
  PHash* = uint64 ##\
  ## PHash is a 64-bit hash, so it's just a synonym for `uint64`.
  ## DCT hash, described in http://phash.org/docs/pubs/thesis_zauner.pdf
  
  MHash* = array[0..8, uint64] ##\
  ## MHash is a 576-bit hash, it's represented as an array of 9 `uint64` values.
  ## Marr hash, used by pHash library (http://phash.org). Very slow to compute and does not give too reliable results (possibly this is due some issues with this implementation).
  
  RHash* = array[0..4, uint64] ##\
  ## RHash is a 320-bit hash, it's represented as an array of 5 `uint64` values.
  ## Radial hash, used by pHash library (http://phash.org). Very slow to compare. 

type 
  Matrix2[T] = object
    width, height: int
    data: seq[T]
  Matrix2f = Matrix2[float32]

proc `[]=`[T] (m: var Matrix2[T], x, y: int, value: T) =
  assert x in 0..<m.width and y in 0..<m.height
  m.data[y * m.width + x] = value

proc `[]`[T] (m: Matrix2[T], x, y: int): T =
  assert x in 0..<m.width and y in 0..<m.height
  m.data[y * m.width + x]

proc `$`*(hash: MHash): string =
  ## Converts MHash to string (hexadecimal) representation.
  for i in 0..8:
    if i > 0:
      result &= ""
    result &= fmt"{hash[i]:016X}"

proc `$`*(hash: RHash): string =
  ## Converts RHash to string (hexadecimal) representation.
  for i in 0..4:
    if i > 0:
      result &= ""
    result &= fmt"{hash[i]:016X}"

proc minmax[T] (m: Matrix2[T]): tuple[min, max: T] =
  if m.width == 0 or m.height == 0:
    return (T(NaN), T(NaN))
  result.min = m.data[0]
  result.max = m.data[0]
  for v in m.data:
    result.min = min(result.min, v)
    result.max = max(result.max, v)

proc initMatrix2[T] (width: int, height: int = -1): Matrix2[T] =
  let height = (if height == -1: width else: height)
  Matrix2[T](width: width, height: height, data: newSeq[T](width * height))

proc initMatrix2f (width: int, height: int = -1): Matrix2f = initMatrix2[float32](width, height)

# Luminosity func from chroma
proc lum(C: Color): float32 {.inline.} =
  0.3 * C.r + 0.59 * C.g + 0.11 * C.b

proc image(matrix: Matrix2f): Image =
  result = newImage(matrix.width, matrix.height)
  for y in 0..<matrix.height:
    for x in 0..<matrix.width:
      result.data[y * matrix.width + x] = ColorHSL(h: 0.0, s: 0.0, l: float(matrix[x, y]) * 100.0).asRgbx

proc grayscale(image: Image): Matrix2f =
  result = initMatrix2f(image.width, image.height)
  for i, pixel in image.data:
    result.data[i] = lum(pixel.color)

# Difference hash; based on https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
proc dhash*(image: Image): uint64 =
  ## Computes DHash of a given image.
  let size = 8
  let width = size + 1
  let grays = image.resize(width, width).grayscale()
  for y in 0..<size:
    for x in 0..<size:
      var offset = y * width + x
      let next = (if ((x + y) and 1) == 0: (offset + 1) else: (offset + width))
      result = (result shl 1) or uint64(grays.data[offset] < grays.data[next])

proc dhashImg*(image: Image): Image =
  ## Converts an image to a debug representation of DHash. Do not use in production.
  let size = 8
  let width = size + 1
  var grays = image.resize(width, width).grayscale()
  result = newImage(size, size)
  for y in 0..<size:
    for x in 0..<size:
      var offset = y * width + x
      let next = (if ((x + y) and 1) == 0: (offset + 1) else: (offset + width))
      var val = uint8(if grays.data[offset] < grays.data[next]: 255 else: 0)
      result.data[y * size + x] = ColorRGBX(r: val, g: val, b: val, a: 255)

proc hhash*(image: Image, lduplets: int = 7): HHash =
  ## Computes HHash of a given image.
  ## 
  ## `lduplets` is a number between 0 and 16, and it can be used to control the ratio of luminosity/color information stored in hash: 
  ##  * 0 = no bits used for storing luminosity information, 64 bits used for storing color information
  ##  * 7 (default) = 28 (7x4) bits used for storing luminosity information, 36 (64-28) bits used for storing color information
  ##  * 16 = 64 (16x4) bits used for storing luminosity information, no bits used for storing color information
  
  let lbucketCnt = lduplets + 1
  let hbucketCnt = 17 - lduplets
  var lumBuckets: array[0..17, float]
  var hueBuckets: array[0..17, float]
  let thumbSize = 256
  let thumb = image.resize(thumbSize, thumbSize)
  var minLum = 1.0
  var maxLum = 0.0
  var minSat = 100.0
  var maxSat = 0.0
  for pixel in thumb.data:
    let l = lum(pixel.color)
    let hsl = pixel.color.hsl
    minLum = min(minLum, l)
    maxLum = max(maxLum, l)
    minSat = min(minSat, hsl.s)
    maxSat = max(maxSat, hsl.s)

  for pixel in thumb.data:
    let hsl = pixel.color.hsl
    let lum = (lum(pixel.color) - minLum) / (maxLum - minLum)
    let lf = min(float(lbucketCnt - 1), lum * float(lbucketCnt))
    let li = int(lf)
    let lt = lf - float(li) # 0..1
    let sat = hsl.s / 100.0 # (hsl.s - minSat) / (maxSat - minSat)
    let st = min(1.0, max(0.0, (sat - 0.15) / (0.5 - 0.15))) # remap saturation to 0..1
    let bt = min(1.0, max(0.0, ((0.5 - abs(lum - 0.5)) - 0.15) / (0.25 - 0.15))) # remap luminance to 0..1
    let hf = min(float(hbucketCnt - 2), hsl.h * float(hbucketCnt - 1) / 360.0)
    let hi = int(hf)
    let ht = hf - float(hi) # 0..1
    #let h = (if sat < 0.15 or lum < 0.15 or lum > 0.85: hbucketCnt - 1 else: hi) # grays have separate bucket

    lumBuckets[li] += (if (li == 0 and lt < 0.5) or (li == lbucketCnt - 1 and lt > 0.5): 1.0 else: (1.0 - abs(lt - 0.5)))
    if li > 0 and lt < 0.5:
      lumBuckets[li - 1] += 0.5 - lt
    if li < lbucketCnt - 1 and lt > 0.5:
      lumBuckets[li + 1] += lt - 0.5

    let k = st * bt # 0..1 (0 = gray colors; 1 = color)
    #echo "st=", st, "; bt=", bt, "; lum=", lum
    hueBuckets[hbucketCnt - 1] += 1.0 - k # gray part goes to gray bucket
    hueBuckets[hi] += k * (1.0 - abs(ht - 0.5))
    if ht < 0.5:
      hueBuckets[(hi + hbucketCnt - 2) mod (hbucketCnt - 1)] += k * (0.5 - ht)
    else:
      hueBuckets[(hi + 1) mod (hbucketCnt - 1)] += k * (ht - 0.5)

  #echo "Luminance buckets: ", lumBuckets
  #echo "Hue buckets: ", hueBuckets

  var lmax = 0.0
  for i in 0..<lbucketCnt:
    lmax = max(lmax, lumBuckets[i])

  var hmax = 0.0
  hueBuckets[hbucketCnt - 1] = hueBuckets[hbucketCnt - 1] / 1.5
  for i in 0..<hbucketCnt:
    hmax = max(hmax, hueBuckets[i])

  #var tmp = newSeq[int]()
  var hash = 0'u64
  for i in 0..<(lbucketCnt - 1):
    let v = uint64(round(15.0 * float(lumBuckets[i]) / float(lmax)))
    #tmp &= int(v)
    hash = (hash shl 4) or v
  #echo "LUM: ", tmp

  #tmp = newSeq[int]()
  for i in 0..<(hbucketCnt - 1):
    let v = uint64(round(15.0 * float(hueBuckets[i]) / float(hmax)))
    #tmp &= int(v)
    hash = (hash shl 4) or v
  #echo "HUE: ", tmp
  return HHash(hash)

proc hhashImgs*(image: Image): tuple[lum, hue: Image] =
  ## Converts an image to a debug representation of HHash. Do not use in production.
  const lbucketCnt = 8
  const hbucketCnt = 10
  let thumbSize = 256
  let thumb = image.resize(thumbSize, thumbSize)
  result.lum = newImage(thumbSize, thumbSize)
  result.hue = newImage(thumbSize, thumbSize)

  var minLum = 1.0
  var maxLum = 0.0
  var minSat = 100.0
  var maxSat = 0.0
  for pixel in thumb.data:
    let l = lum(pixel.color)
    let hsl = pixel.color.hsl
    minLum = min(minLum, l)
    maxLum = max(maxLum, l)
    minSat = min(minSat, hsl.s)
    maxSat = max(maxSat, hsl.s)

  for i, pixel in thumb.data:
    let hsl = pixel.color.hsl
    let lum = (lum(pixel.color) - minLum) / (maxLum - minLum)
    let l = int(min(lbucketCnt - 1, lum * lbucketCnt))
    let sat = (hsl.s - minSat) / (maxSat - minSat)
    let h = (if sat < 0.2 or lum < 0.15 or lum > 0.85: hbucketCnt - 1 else: int(min(hbucketCnt - 2, hsl.h * (hbucketCnt - 1) / 360.0))) # grays have separate bucket
    
    result.lum.data[i] = ColorHSL(h: 0.0, s: 0.0, l: float32(l) * 100.0 / float32(lbucketCnt - 1)).asRgbx
    result.hue.data[i] = (if h == hbucketCnt - 1:
      ColorHSL(h: 0.0, s: 0.0, l: 50.0) else:
      ColorHSL(h: float32(h) * 360.0 / float32(hbucketCnt - 2), s: 100.0, l: 50.0)).asRgbx 

proc meanKernel(size: int = 7): Matrix2f =
  result = initMatrix2f(size)
  for v in result.data.mitems:
    v = 1.0

proc correlate(image: Matrix2f, kernel: Matrix2f): Matrix2f =
  result = initMatrix2f(image.width, image.height)
  let mx1 = kernel.width div 2
  let my1 = kernel.height div 2
  let mx2 = kernel.width - mx1 - 1
  let my2 = kernel.height - my1 - 1
  let mxe = image.width - mx2
  let mye = image.height - my2

  # Compute inner pixels
  for y in my1..<mye:
    for x in mx1..<mxe:
      var value = 0.0
      for ym in -my1..my2:
        for xm in -mx1..mx2:
          value += image[x + xm, y + ym] * kernel[mx1 + xm, my1 + ym]
      result[x, y] = value

  # Compute outer pixels
  for y in 0..<image.height:
    var x = 0
    while x < image.width:
      var value = 0.0
      for ym in -my1..my2:
        for xm in -mx1..mx2:
          if ((x + xm) in 0..<image.width) and ((y + ym) in 0..<image.height):
            value += image[x + xm, y + ym] * kernel[mx1 + xm, my1 + ym]
      result[x, y] = value

      if (y < my1) or (y >= mye) or (x < mx1 - 1) or (x >= mxe):
        x += 1
      else:
        x = mxe

const dctCoeffs = block:
  var result: array[1..8, array[1..8, array[0..31, array[0..31, float]]]]
  for i in 1..8:
    for j in 1..8:
      for k in 0..<32:
        for l in 0..<32:
          result[i][j][k][l] = cos((2 * float(k) + 1) * float(i) * PI / 64) * cos((2 * float(l) + 1) * float(j) * PI / 64)
  result

proc phash*(image: Image): uint64 =
  ## Computes PHash of a given image.
  
  const sz = 32

  let kernel = meanKernel()
  var gray = image.resize(sz, sz).grayscale()
  gray = gray.correlate(kernel)
  
  # by-element computation of DCT
  # TODO: compare with using matrices and FFT
  var dct: array[1..8, array[1..8, float32]]
  var list: array[0..63, float32]
  for i in 1..8:
    for j in 1..8:
      var sum = 0.0
      for k in 0..<sz:
        for l in 0..<sz:
          sum += gray[k, l] * dctCoeffs[i][j][k][l]
      dct[i][j] = sum
      list[(i - 1) * 8 + (j - 1)] = dct[i][j]

  sort(list)
  let mdn = (list[31] + list[32]) * 0.5 # compute median

  for i in 1..8:
    for j in 1..8:
      result = (result shl 1) or uint64(dct[i][j] > mdn)

proc phashImg*(image: Image): Image =
  ## Converts an image to a debug representation of PHash. Do not use in production.
  
  const sz = 32
  let thumb = image.resize(sz, sz)
  var mat: array[0..(sz - 1), array[0..(sz - 1), float32]]
  var minLum = 1.0
  var maxLum = 0.0
  for y in 0..<sz:
    for x in 0..<sz:
      var offset = y * sz + x
      mat[x][y] = lum(thumb.data[offset].color)
      minLum = min(minLum, mat[x][y])
      maxLum = max(maxLum, mat[x][y])

  for y in 0..<sz:
    for x in 0..<sz:
      mat[x][y] = 256.0 * (mat[x][y] - minLum) / (maxLum - minLum) - 128.0
  
  var dct: array[0..(sz - 1), array[0..(sz - 1), float32]]
  var avg = 0.0
  for i in 0..<sz:
    for j in 0..<sz:
      let ci = (if i == 0: 1.0 / sqrt(float(sz)) else: sqrt(2.0 / float(sz)))
      let cj = (if j == 0: 1.0 / sqrt(float(sz)) else: sqrt(2.0 / float(sz)))
      var sum = 0.0
      for k in 0..<sz:
        for l in 0..<sz:
          sum += mat[k][l] * cos((2 * float(k) + 1) * float(i) * PI / (2 * float(sz))) * cos((2 * float(l) + 1) * float(j) * PI / (2 * float(sz)))
      dct[i][j] = ci * cj * sum
      if i in 1..8 and j in 1..8:
        avg += dct[i][j] / 64.0

  #echo "Original DCT: ", dct
  # Decode back
  for y in 0..<sz:
    for x in 0..<sz:
      mat[x][y] = 0.0
      if x in 1..8 and y in 1..8:
        dct[x][y] = (if dct[x][y] > avg: 20.0 else: -20.0)
      else:
        dct[x][y] = 0.0

  dct[0][0] = 16.0

  for i in 0..<sz:
    for j in 0..<sz:
      var sum = 0.0
      for k in 0..<sz:
        for l in 0..<sz:
          let ci = (if k == 0: 1.0 / sqrt(float(sz)) else: sqrt(2.0 / float(sz)))
          let cj = (if l == 0: 1.0 / sqrt(float(sz)) else: sqrt(2.0 / float(sz)))
          sum += ci * cj * dct[k][l] * cos((2 * float(i) + 1) * float(k) * PI / (2 * float(sz))) * cos((2 * float(j) + 1) * float(l) * PI / (2 * float(sz)))
      mat[i][j] = sum

  #echo "Decoded mat: ", mat
  for y in 0..<sz:
    for x in 0..<sz:
      var offset = y * sz + x
      thumb.data[offset] = ColorHSL(h: 0.0, s: 0.0, l: max(0, min(100.0, mat[x][y] * 100.0 / 256.0 + 50.0))).asRgbx
  return thumb

proc marrKernel(alpha: int = 2, level: int = 1): Matrix2f =
  let sigma = 4 * (alpha ^ level)
  result = initMatrix2f(2 * sigma + 1)
  let k = pow(float(alpha), -float(level))
  for y in 0..<result.height:
    for x in 0..<result.width:
      let ax = float(x - sigma)
      let ay = float(y - sigma)
      let a = k * k * (ax * ax + ay * ay)
      result[x, y] = (2 - a) * exp(-a / 2)

const marrKernel8x8 = marrKernel()

proc histogram(image: Matrix2f, levels: int, min, max: float32): seq[int] =
  result = newSeq[int](levels)
  for v in image.data:
    if v in min..max:
      result[min(levels - 1, int((v - min) * float(levels) / (max - min)))] += 1

# Applies Images Histogram Equalization algorithm
# (https://en.wikipedia.org/wiki/Histogram_equalization)
proc equalize(image: var Matrix2f, levels: int = 256) =
  var (min, max) = image.minmax
  var hist = image.histogram(levels, min, max)
  for i in 1..<hist.len:
    hist[i] += hist[i - 1]
  let total = float(if hist.len == 0 or hist[^1] == 0: 1 else: hist[^1])
  for p in image.data.mitems:
    let pos = int((p - min) * float(levels - 1) / (max - min))
    if pos in 0..<levels:
      p = min + (max - min) * float(hist[pos]) / total

proc normalize(matrix: var Matrix2f) =
  if matrix.width == 0 or matrix.height == 0:
    return
  var (min, max) = matrix.minmax
  for v in matrix.data.mitems:
    v = (v - min) / (max - min)

# Based on https://github.com/poslegm/scala-phash/blob/master/src/main/scala/scalaphash/PhashInternal.scala
proc mhash*(image: Image, alpha: int = 2, level: int = 1): MHash =
  ## Computes MHash of a given image.
  
  image.blur(float(max(image.width, image.height)) / 200.0)
  var gray = image.resize(512, 512).grayscale()
  gray.equalize(256)

  let kernel = (if alpha == 2 and level == 1: marrKernel8x8 else: marrKernel(alpha, level))
  var corr = gray.correlate(kernel)
  corr.normalize()

  var blocks = initMatrix2f(31)
  for y in 0..<31:
    for x in 0..<31:
      for j in 0..<16:
        for i in 0..<16:
          blocks[x, y] = blocks[x, y] + corr[x * 16 + i, y * 16 + j]
  
  var i = 0
  var bits = 0
  var hash: array[0..8, uint64]
  for y0 in countup(0, 28, 4):
    for x0 in countup(0, 28, 4):
      var avg = 0.0
      for x in x0..(x0 + 2):
        for y in y0..(y0 + 2):
          avg += blocks[x, y]
      avg /= 9.0
      for x in x0..(x0 + 2):
        for y in y0..(y0 + 2):
          #echo "avg = ", avg, ", blocks[x, y] = ", blocks[x, y]
          hash[i] = (hash[i] shl 1) or uint64(blocks[x, y] > avg)
          bits += 1
          if bits == 64:
            bits = 0
            i += 1
  return MHash(hash)

const projectionsCount = 180
const theta180 = block:
  var result: array[0..(projectionsCount - 1), float]
  for i in 0..<projectionsCount:
    result[i] = float(i) * PI / float(projectionsCount)
  result
const tanTheta180 = block:
  var result: array[0..(projectionsCount - 1), float]
  for i in 0..<projectionsCount:
    result[i] = tan(theta180[i])
  result

# Based on https://github.com/poslegm/scala-phash/blob/master/src/main/scala/scalaphash/PhashInternal.scala
proc rhash*(image: Image): RHash =
  ## Computes RHash of a given image.
  
  const size = 128
  image.blur(float(max(image.width, image.height)) / 200.0)
  var gray = image.resize(size, size).grayscale()

  let xOff = (gray.width shr 1) + (gray.width and 1)
  let yOff = (gray.height shr 1) + (gray.height and 1)
  var projections: array[0..(projectionsCount - 1), seq[float32]]
  for i in 0..(projectionsCount - 1):
    projections[i] = newSeq[float32]()
  
  # Compute first quarter
  for k in 0..(projectionsCount div 4):
    let alpha = tanTheta180[k]
    for x in 0..<size:
      let y = alpha * float(x - xOff)
      let yd = int(floor(y + (if y >= 0: 0.5 else: -0.5)))
      if (yd + yOff) in 0..<size:
        projections[k] &= gray[x, yd + yOff]
      if ((yd + xOff) in 0..<size) and (k != projectionsCount div 4):
        projections[(projectionsCount div 2) - k] &= gray[yd + xOff, x]

  # Compute last quarter
  var j = 0
  for k in (3 * projectionsCount div 4)..<projectionsCount:
    let alpha = tanTheta180[k]
    for x in 0..<size:
      let y = alpha * float(x - xOff)
      let yd = int(floor(y + (if y >= 0: 0.5 else: -0.5)))
      if (yd + yOff) in 0..<size:
        projections[k] &= gray[x, yd + yOff]
      if ((yOff - yd) in 0..<size) and ((2 * yOff - x) in 0..<size) and (k != 3 * projectionsCount div 4):
        projections[k - j] &= gray[-yd + yOff, -(x - yOff) + yOff]
    j += 2

  # Calculate features
  var features: array[0..(projectionsCount - 1), float]
  var featuresSum, featuresSumSq = 0.0
  for k in 0..<projectionsCount:
    var lineSum, lineSumSq = 0.0
    for v in projections[k]:
      lineSum += v
      lineSumSq += v * v
    let count = float(projections[k].len)
    features[k] = lineSumSq / count - lineSum * lineSum / (count * count)
    featuresSum += features[k]
    featuresSumSq += features[k] * features[k]
  
  let mean = featuresSum / float(projectionsCount)
  let meanSq = mean * mean
  let x = sqrt(featuresSumSq / float(projectionsCount) - meanSq)
  for f in features.mitems:
    f = (f - mean) / x

  # Calculate radial hash
  const coeffsCount = 40

  var max = 0.0
  var min = 1e100
  var digest: array[0..(coeffsCount - 1), float]
  for i in 0..<coeffsCount:
    var sum = 0.0
    for j, f in features:
      sum += f * cos(PI * float(2 * j + 1) * float(i) / float(2 * features.len))
    let value = (if i == 0: sum / sqrt(float(features.len)) else: (sum * sqrt(2.0 / float(features.len))))
    max = max(max, value)
    min = min(min, value)
    digest[i] = value

  for i in 0..<5:
    for j in 0..<8:
      let index = i * 8 + j
      result[i] = (result[i] shl 8) or uint64(255.0 * (digest[index] - min) / (max - min))

proc dhash*(path: string): DHash = dhash(readImage(path))
  ## Computes DHash of an image loaded from the given file path.
proc rhash*(path: string): RHash = rhash(readImage(path))
  ## Computes RHash of an image loaded from the given file path.
proc mhash*(path: string): MHash = mhash(readImage(path))
  ## Computes MHash of an image loaded from the given file path.
proc phash*(path: string): PHash = phash(readImage(path))
  ## Computes PHash of an image loaded from the given file path.
proc hhash*(path: string): HHash = hhash(readImage(path))
  ## Computes HHash of an image loaded from the given file path.

# Compare hashes

proc diff*(hash1, hash2: uint64): int =
  ## Compute the difference (Hamming distance) between simple 64-bit hashes (like DHash and PHash). Uses `popcount` instruction. The result is between 0 (identical images) and 64.
  ## Usually even the completely different images produce result in 40..50 range and don't go up all the way to 64.
  popcount(hash1 xor hash2)

proc diff*(hash1, hash2: HHash): int =
  ## Compute the difference (Hamming distance) between HHashes. The result is between 0 (identical images) and 240.
  ## Usually even the completely different images produce result in 80..100 range and don't go up all the way to 240.
  ## Unfortunately, fast `popcount` won't work here (because histogram hash is actually an array of 16 four-bit integers)
  var h1 = uint64(hash1)
  var h2 = uint64(hash2)
  for i in 0..<16:
    result += abs(int(h1 and 15) - int(h2 and 15))
    h1 = h1 shr 4
    h2 = h2 shr 4

proc diff*(hash1, hash2: MHash): int =
  ## Compute the difference (Hamming distance) between MHashes. The result is between 0 (identical images) and 576.
  ## Usually even the completely different images produce result in 270..350 range and don't go up all the way to 576.
  for i in 0..8:
    result += popcount(hash1[i] xor hash2[i])

proc diff*(hash1, hash2: RHash): float =
  ## Compute the difference (Hamming distance) between RHashes. Notice that the result is a floating-point value (between 0.0 and 100.0), not an integer.
  ## Usually even the completely different images produce result in 75..85 range.
  var mean1, mean2 = 0.0
  for i in 0..<5:
    for j in 0..<8:
      mean1 += float((hash1[i] shr (8 * j)) and 0xff)
      mean2 += float((hash2[i] shr (8 * j)) and 0xff)
  mean1 /= 40.0
  mean2 /= 40.0
  #echo mean1, "; ", mean2
  var max = 0.0
  for k in 0..<40:
    var num = 0.0
    var den1, den2 = 0.0
    var buf = newSeq[string]()
    for i in 0..<5:
      for j in 0..<8:
        var index1 = i * 8 + j
        let index2 = (40 + index1 - k) mod 40
        var h1 = float((hash1[i] shr (8 * j)) and 0xff)
        var h2 = float((hash2[index2 div 8] shr (8 * (index2 mod 8))) and 0xff)
        let d1 = h1 - mean1
        let d2 = h2 - mean2
        num += d1 * d2
        den1 += d1 * d1
        den2 += d2 * d2
        buf &= fmt"[{index1}, {index2}, {h1}, {h2}, {d1}, {d2}]"
    #echo buf
    max = max(max, num / sqrt(den1 * den2))
  return (1.0 - max) * 100.0