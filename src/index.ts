import { range as _range } from './utils'

type Shape = number[]

type Operand = number | NDArray

type BinaryOperator = (n: number, m: number) => number

type UniversalBinaryOperator = (a: Operand, b: Operand) => NDArray

type UnaryOperator = (n: number) => number

type UniversalUnaryOperator = (n: Operand) => NDArray

interface Range {
  start?: number
  end?: number
  step?: number
}

export const All: Range = Object.freeze({})

export function range (start?: number, end?: number, step?: number): Range {
  if (start != null && end == null) {
    end = start
    start = 0
  }
  if (step == null) {
    step = 1
  }
  return { start, end, step }
}

function* enumerateRange (range: Range, size: number): Iterable<number> {
  let { start, end, step } = range
  if (step == null) {
    step = 1
  }
  if (start == null) {
    if (step > 0) {
      start = -Infinity
    } else {
      start = Infinity
    }
  } else if (start < 0) {
    start = size + start
  }
  start = Math.floor(start)
  if (end == null) {
    if (step > 0) {
      end = Infinity
    } else {
      end = -Infinity
    }
  } else if (end < 0) {
    end = size + end
  }
  end = Math.floor(end)
  step = Math.floor(step)
  if (step === 0) {
    throw new Error('Step cannot be 0')
  }

  if (step > 0) {
    for (let i = Math.max(start, 0); i < end && i < size; i += step) {
      yield i
    }
  } else {
    for (let i = Math.min(start, size - 1); i > end && i >= 0; i += step) {
      yield i
    }
  }
}

function enumerateRanges (ranges: Array<Range>, shape: Shape): Iterable<[number[], number[]]> {
  if (ranges.length !== shape.length) {
    throw new Error('Sizes of ranges and shape are different')
  }
  if (ranges.length === 0) {
    return
  }

  return ranges.reduce(function* (a: Iterable<[number[], number[]]>, r: Range, i): Iterable<[number[], number[]]> {
    for (const [idx1, idx2] of a) {
      let k = 0
      for (const j of enumerateRange(r, shape[i])) {
        yield [idx1.concat([j]), idx2.concat([k])]
        k++
      }
    }
  }, [[[], []]])
}

function countRange (r: Range, size: number) {
  let c = 0
  for (const i of enumerateRange(r, size)) {
    c++
  }
  return c
}

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
      axes = _range(this.shape.length).reverse()
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
    const i = _range(this.shape.length)
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

  slice (...indexOrRanges: Array<number | Range>): NDArray {
    if (indexOrRanges.length > this.shape.length) {
      throw new Error('Too many indices')
    }
    while (indexOrRanges.length < this.shape.length) {
      indexOrRanges.push(All)
    }

    const ranges = indexOrRanges.map((ir) => {
      if (typeof ir === 'number') {
        return range(ir, ir + 1)
      }
      return ir
    })
    const resultShape = ranges.map((r, i) => countRange(r, this.shape[i]))
    const result = zeros(resultShape)

    if (result.size > 0) {
      for (const [idx1, idx2] of enumerateRanges(ranges, this.shape)) {
        result.set(idx2, this.get(idx1))
      }
    }

    const normalizedShape = resultShape.filter((s, i) => typeof indexOrRanges[i] !== 'number')
    return result.reshape(normalizedShape)
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
      const s = shape[j]
      const m = indices[j] = k % s
      k = (k - m) / s
    }
    indices[0] = k
    yield indices
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
  const ai = indexIdLists.map((ids) => new Array(ids.length) as number[])
  for (const idx of enumerateIndices(dims)) {
    const ri = resultShape.length > 0 ? resultIndexIds.map((i) => idx[i]) : [0]
    let p = 1
    for (let i = 0; i < arrays.length; i++) {
      for (let j = 0; j < indexIdLists[i].length; j++) {
        ai[i][j] = idx[indexIdLists[i][j]]
      }
      p *= arrays[i].get(ai[i])
    }
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
  let i = 0
  for (const idx of enumerateIndices(resultShape)) {
    for (let j = 0; j < r; j++) {
      ia[j] = idx[j] * ma[j]
      ib[j] = idx[j] * mb[j]
    }
    result.data[i] = f(a.get(ia), b.get(ib))
    i++
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
