function assert (x: boolean) {
  if (!x) {
    throw new Error('assertion failed')
  }
}

class NDArray {
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
    assert(indices.length === this.shape.length)

    let idx = indices[0]
    for (let i = 1; i < this.shape.length; i++) {
      idx = idx * this.shape[i - 1] + indices[i]
    }

    return this.data[idx]
  }

  set (indices: number[], value: number) {
    //
  }
}

function createArray (raw: any[], shape?: number[]): NDArray {
  return new NDArray([], [])
}

function subscript (array: NDArray, indices: number[]): NDArray {
  assert(indices.length <= array.shape.length)

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

const a = new NDArray([1,2,3,4,5,6,7,8], [2,2,2])
