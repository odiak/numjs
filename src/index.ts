import { range } from './utils'

type Shape = number[]

type Operand = number | NDArray

type BinaryOperator = (n: number, m: number) => number

type UniversalBinaryOperator = (a: Operand, b: Operand) => NDArray

type UnaryOperator = (n: number) => number

type UniversalUnaryOperator = (n: Operand) => NDArray

export class NDArray {
  data: number[]
  shape: number[]
  size: number

  constructor (data: number[], shape: Shape) {
    if (data.length !== shapeProduct(shape)) {
      throw new Error('invalid array and shape')
    }
    this.data = data
    this.shape = shape
    this.size = shapeProduct(shape)
  }

  static empty (shape: number[] | number): NDArray {
    return new NDArray([], [])
  }

  get (indices: (number | number[]) = 0): number {
    if (typeof indices === 'number') {
      indices = [indices]
    }
    const idx = flattenIndices(indices, this.shape)
    return this.data[idx]
  }

  set (indices: (number | number[]), value: number) {
    if (typeof indices === 'number') {
      indices = [indices]
    }
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

  transpose (axes?: number[]): NDArray {
    if (axes) {
      if (axes.length !== this.shape.length) {
        throw new Error('Invalid axes')
      }
      for (let i = 0; i < this.shape.length; i++) {
        if (!axes.includes(i)) {
          throw new Error('Invalid axes')
        }
      }
    } else {
      axes = range(this.shape.length).reverse()
    }
    const resultShape = axes.map((s) => this.shape[s])
    const newArray = zeros(resultShape)
    for (let idx of enumerateIndices(this.shape)) {
      const resultIndex = axes.map((s) => idx[s])
      newArray.set(resultIndex, this.get(idx))
    }
    return newArray
  }

  swapAxes (a1: number, a2: number): NDArray {
    if (a1 < 0 || a1 >= this.shape.length) {
      throw new Error('Invalid axis 1')
    }
    if (a2 < 0 || a2 >= this.shape.length) {
      throw new Error('Invalid axis 2')
    }
    const i = range(this.shape.length)
    i[a1] = a2
    i[a2] = a1
    return this.transpose(i)
  }

  add (x: Operand) {
    return add(this, x)
  }
  sub (x: Operand) {
    return sub(this, x)
  }
  mul (x: Operand) {
    return mul(this, x)
  }
  div (x: Operand) {
    return div(this, x)
  }
  pow (x: Operand) {
    return pow(this, x)
  }
  neg () {
    return neg(this)
  }
  argMin (axis: number) {
    return argMin(this, axis)
  }
  argMax (axis: number) {
    return argMax(this, axis)
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

export function zerosLike (array: NDArray): NDArray {
  return zeros(array.shape)
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

function operate (f: BinaryOperator, a: Operand, b: Operand): NDArray {
  if (typeof a === 'number') {
    if (typeof b === 'number') {
      a = createArray([a])
    } else {
      a = createArray([a]).reshape((b as NDArray).shape.map(() => 1))
    }
  }
  if (typeof b === 'number') {
    b = createArray([b]).reshape(a.shape.map(() => 1))
  }

  if (a.shape.length !== b.shape.length) {
    throw new Error('Incompatible shape')
  }
  const r = a.shape.length
  for (let i = 0; i < r; i++) {
    if (a.shape[i] !== 1 && b.shape[i] !== 1 && a.shape[i] !== b.shape[i]) {
      throw new Error('Incompatible shape')
    }
  }

  const ma = a.shape.map((s) => s > 1 ? 1 : 0)
  const mb = b.shape.map((s) => s > 1 ? 1 : 0)

  const resultShape = a.shape.map((s, i) => Math.max(s, (b as NDArray).shape[i]))
  const result = zeros(resultShape)
  const ia = new Array(r)
  const ib = new Array(r)
  for (const i of enumerateIndices(resultShape)) {
    for (let j = 0; j < r; j++) {
      ia[j] = i[j] * ma[j]
      ib[j] = i[j] * mb[j]
    }
    result.set(i, f(a.get(ia), b.get(ib)))
  }
  return result
}

function createUniversalBinaryFunction (f: BinaryOperator): UniversalBinaryOperator {
  return (a: Operand, b: Operand) => operate(f, a, b)
}

export const add = createUniversalBinaryFunction((a, b) => a + b)
export const sub = createUniversalBinaryFunction((a, b) => a - b)
export const mul = createUniversalBinaryFunction((a, b) => a * b)
export const div = createUniversalBinaryFunction((a, b) => a / b)
export const pow = createUniversalBinaryFunction((a, b) => a ** b)

function operateUnary (f: UnaryOperator, a: Operand): NDArray {
  if (typeof a === 'number') {
    a = createArray([a])
  }

  const result = zerosLike(a)
  for (const i of enumerateIndices(result.shape)) {
    result.set(i, f(a.get(i)))
  }

  return result
}

function createUniversalUnaryOperator (f: UnaryOperator): UniversalUnaryOperator {
  return (a: Operand) => operateUnary(f, a)
}

export const neg = createUniversalUnaryOperator((a) => -a)
export const exp = createUniversalUnaryOperator((a) => Math.exp(a))

export function argMin (array: NDArray, axis: number): NDArray {
  if (axis < 0 || axis >= array.shape.length) {
    throw new Error('invalid axis')
  }

  const resultShape = array.shape.slice()
  resultShape.splice(axis, 1)

  if (shapeProduct(array.shape) === 0) {
    return new NDArray([], resultShape)
  }

  const shape = array.shape
  const subShape = shape.slice()
  subShape[axis] = 1

  const result = zeros(resultShape)

  for (const i of enumerateIndices(subShape)) {
    let min = array.get(i)
    let minIndex = 0
    for (let j = 1; j < shape[axis]; j++) {
      i[axis] = j
      const v = array.get(i)
      if (v < min) {
        min = v
        minIndex = j
      }
    }
    i.splice(axis, 1)
    result.set(i, minIndex)
  }

  return result
}

export function argMax (array: NDArray, axis: number): NDArray {
  if (axis < 0 || axis >= array.shape.length) {
    throw new Error('invalid axis')
  }

  const resultShape = array.shape.slice()
  resultShape.splice(axis, 1)

  if (shapeProduct(array.shape) === 0) {
    return new NDArray([], resultShape)
  }

  const shape = array.shape
  const subShape = shape.slice()
  subShape[axis] = 1

  const result = zeros(resultShape)

  for (const i of enumerateIndices(subShape)) {
    let max = array.get(i)
    let maxIndex = 0
    for (let j = 1; j < shape[axis]; j++) {
      i[axis] = j
      const v = array.get(i)
      if (v > max) {
        max = v
        maxIndex = j
      }
    }
    i.splice(axis, 1)
    result.set(i, maxIndex)
  }

  return result
}
