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

export function einsum ([i1, i2, i3]: [string[], string[], string[]], a: NDArray, b: NDArray): NDArray {
  if (i1.length !== a.shape.length) {
    throw new Error('ah')
  }
  if (i2.length !== b.shape.length) {
    throw new Error('ah')
  }
  const dimByIndexName: {[s: string]: number} = {}
  for (const [i, iName] of i1.entries()) {
    if (iName in dimByIndexName) {
      if (dimByIndexName[iName] !== a.shape[i]) {
        throw new Error('ah')
      }
    } else {
      dimByIndexName[iName] = a.shape[i]
    }
  }
  for (const [i, iName] of i2.entries()) {
    if (iName in dimByIndexName) {
      if (dimByIndexName[iName] !== b.shape[i]) {
        throw new Error('ah')
      }
    } else {
      dimByIndexName[iName] = b.shape[i]
    }
  }
  const resultShape: Shape = []
  const resultIndexByName: {[s: string]: number} = {}
  for (const iName of i3) {
    if (!(iName in dimByIndexName)) {
      throw new Error('ah')
    }
    resultShape.push(dimByIndexName[iName])
    resultIndexByName[iName] = resultShape.length - 1
  }
  const restIndexNames: string[] = []
  const restShape: Shape = []
  const restIndexByName: {[s: string]: number} = {}
  for (const [iName, d] of Object.entries(dimByIndexName)) {
    if (!i3.includes(iName)) {
      restIndexNames.push(iName)
      restShape.push(d)
      restIndexByName[iName] = restIndexNames.length - 1
    }
  }

  const result = zeros(resultShape)
  for (const r of enumerateIndices(resultShape)) {
    let sum = 0
    const ia: Shape = i1.map(() => 0)
    const ib: Shape = i2.map(() => 0)
    for (const [i, iName] of i1.entries()) {
      if (iName in resultIndexByName) {
        ia[i] = r[resultIndexByName[iName]]
      }
    }
    for (const [i, iName] of i2.entries()) {
      if (iName in resultIndexByName) {
        ib[i] = r[resultIndexByName[iName]]
      }
    }
    for (const s of enumerateIndices(restShape)) {
      for (const [i, iName] of i1.entries()) {
        if (iName in restIndexByName) {
          ia[i] = s[restIndexByName[iName]]
        }
      }
      for (const [i, iName] of i2.entries()) {
        if (iName in restIndexByName) {
          ib[i] = s[restIndexByName[iName]]
        }
      }
      sum += a.get(ia) * b.get(ib)
    }
    result.set(r, sum)
  }

  return result
}
