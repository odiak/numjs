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
    if (!isReshapable(this.shape, shape)) {
      throw new Error('incompatible shape')
    }
    return new NDArray(this.data, shape)
  }
}

function isValidShape (shape: number[]): boolean {
  return shape.every((n) => Number.isFinite(n) && n >= 0)
}

function isReshapable (oldShape: number[], newShape: number[]): boolean {
  return isValidShape(oldShape) && isValidShape(newShape) && oldShape.reduce((a, b) => a * b, 1) === newShape.reduce((a, b) => a * b, 1)
}

function createArray (raw: any[], shape?: number[]): NDArray {
  return new NDArray([], [])
}

function flattenIndices (indices: number[], shape: number[]): number {
  let idx = indices[0]
  for (let i = 1; i < shape.length; i++) {
    idx = idx * shape[i - 1] + indices[i]
  }
  return idx
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
