type Shape = number[]

export class NDArray {
  data: number[]
  shape: number[]

  constructor (data: number[], shape: Shape) {
    if (data.length !== shapeProduct(shape)) {
      throw new Error('invalid array and shape')
    }
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
    for (let idx of enumerateIndices(newArray.shape)) {
      newArray.set(idx, this.get(idx.slice().reverse()))
    }
    return newArray
  }
}

export function zeros (shapeOrNumber: Shape | number): NDArray {
  const shape = typeof shapeOrNumber === 'number' ? [shapeOrNumber] : shapeOrNumber
  if (!isValidShape(shape)) {
    throw new Error('invalid shape')
  }
  const p = shapeProduct(shape)
  return new NDArray((new Array(p)).fill(0), shape)
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

function* enumerateIndices (shape: Shape): Iterable<number[]> {
  const p = shapeProduct(shape)
  const n = shape.length
  if (n === 0 || p === 0) {
    return
  }

  const indices = new Array(n)
  for (let i = 0; i < p; i++) {
    let k = i
    for (let j = n - 1; j > 0; j--) {
      indices[j] = k % shape[j]
      k = (k / shape[j]) | 0
    }
    indices[0] = k
    yield indices.slice()
  }
}

export function createArray (raw: any[]): NDArray {
  const shape = []
  for (let a = raw; Array.isArray(a); a = a[0]) {
    shape.push(a.length)
  }
  const data = flatten(raw)
  if (data.length !== shapeProduct(shape) || data.some((x) => typeof x !== 'number')) {
    throw new Error('invalid argument')
  }
  return new NDArray(data, shape)
}

function flatten (array: any[], dest: any[] = []): any[] {
  for (const a of array) {
    if (Array.isArray(a)) {
      flatten(a, dest)
    } else {
      dest.push(a)
    }
  }
  return dest
}

function flattenIndices (indices: number[], shape: number[]): number {
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

export function einsum (indexNameLists: Array<Array<string>>, resultIndexNames: Array<string>, ...arrays: Array<NDArray>): NDArray {
  if (indexNameLists.length === 0) {
    throw new Error('Specify one or more elements for 1st argument')
  }
  for (const [i, indexNames] of indexNameLists.entries()) {
    if (indexNames.length !== arrays[i].shape.length) {
      throw new Error(`Number of index names and rank of array at ${i}`)
    }
  }
  const idByIndexName: {[s: string]: number} = {}
  const dims: Shape = []
  const indexIdLists = indexNameLists.map((a) => a.map((i) => 0))
  for (const [i, indexNames] of indexNameLists.entries()) {
    for (const [j, iName] of indexNames.entries()) {
      if (iName in idByIndexName) {
        if (dims[idByIndexName[iName]] !== arrays[i].shape[j]) {
          throw new Error('shape is not matched')
        }
        indexIdLists[i][j] = idByIndexName[iName]
      } else {
        const id = dims.length
        dims.push(arrays[i].shape[j])
        idByIndexName[iName] = id
        indexIdLists[i][j] = id
      }
    }
  }
  const resultIndexIds: number[] = []
  const resultShape: Shape = []
  for (const iName of resultIndexNames) {
    const id = idByIndexName[iName]
    if (id == null) {
      throw new Error(`Unknown index name '${iName}'`)
    }
    resultIndexIds.push(id)
    resultShape.push(dims[id])
  }

  const result = zeros(resultShape.length > 0 ? resultShape : [1])
  for (const idx of enumerateIndices(dims)) {
    const ri = resultShape.length > 0 ? resultIndexIds.map((i) => idx[i]) : [0]
    const p = indexIdLists
      .map((indexIds, i) => arrays[i].get(indexIds.map((j) => idx[j])))
      .reduce((a, b) => a * b, 1)
    result.set(ri, result.get(ri) + p)
  }

  return result
}
