export class NDArray {
  data: number[]
  shape: number[]

  constructor (data, shape) {
    this.data = data
    this.shape = shape
  }

  static empty (shape: number[] | number): NDArray {
    return new NDArray([], [])
  }

  get (indices: number[]): number {
    const idx = flattenIndices(indices, this.shape)
    return this.data[idx]
  }

  set (indices: number[], value: number) {
    const idx = flattenIndices(indices, this.shape)
    this.data[idx] = value
  }

  reshape (shape: number[]): NDArray {
    const i = shape.indexOf(-1)
    if (i !== -1) {
      const p = shapeProduct(this.shape)
      const q = shapeProduct(shape.filter((n) => n >= 0))
      if (p % q === 0) {
        shape[i] = p / q
      }
    }

    if (!isReshapable(this.shape, shape)) {
      throw new Error('incompatible shape')
    }
    return new NDArray(this.data.slice(), shape)
  }

  transpose (): NDArray {
    const newArray = new NDArray(this.data.slice(), this.shape.slice().reverse())
    for (let idx of enumerateIndices(newArray)) {
      newArray.set(idx, this.get(idx.slice().reverse()))
    }
    return newArray
  }
}

function shapeProduct (indices: number[]): number {
  if (indices.length === 0) {
    return 0
  }
  return indices.reduce((a, b) => a * b, 1)
}

function isValidShape (shape: number[]): boolean {
  return shape.every((n) => Number.isFinite(n) && n >= 0)
}

function isReshapable (oldShape: number[], newShape: number[]): boolean {
  return isValidShape(oldShape) && isValidShape(newShape) && shapeProduct(oldShape) === shapeProduct(newShape)
}

function* enumerateIndices (array: NDArray): Iterable<number[]> {
  const { shape } = array
  if (shape.length === 0 || shapeProduct(shape) === 0) {
    return
  }

  const indices = shape.map(() => 0)
  while (indices.every((idx, i) => idx < shape[i])) {
    yield indices.slice()
    indices[0] += 1
    for (let i = 1; i < indices.length; i++) {
      if (indices[i - 1] < shape[i - 1]) {
        break
      }
      indices[i - 1] = 0
      indices[i] += 1
    }
  }
}

function createArray (raw: any[], shape?: number[]): NDArray {
  return new NDArray([], [])
}

export function flattenIndices (indices: number[], shape: number[]): number {
  const ks = [1]
  let k = 1
  for (let i = shape.length - 1; i >= 1; i--) {
    k *= shape[i]
    ks.unshift(k)
  }
  return indices.reduce((a, idx, i) => a + idx * ks[i], 0)
}

function subscript (array: NDArray, indices: number[]): NDArray {
  while (indices.length < array.shape.length) {
    indices.push(null)
  }

  const newShape = array.shape.slice()
  // for (let [idx, i] of indices.entries()) {
  //   if (idx == null) {
  //     newShape[i] = idx
  //   }
  // }
  const newArray = NDArray.empty(newShape)

  return newArray
}
