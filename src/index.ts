import { range as _range } from './utils'

export type Shape = number[]

export type Operand = number | NDArray

export type BinaryOperator = (n: number, m: number) => number

export type UniversalBinaryOperator = (a: Operand, b: Operand, out?: NDArray) => NDArray

export type UnaryOperator = (n: number) => number

export type UniversalUnaryOperator = (n: Operand, out?: NDArray) => NDArray

export interface Range {
  start?: number
  end?: number
  step?: number
}

export type ShapeOrNumber = Shape | number

export const All: Range = Object.freeze({})

export const NewAxis = null

export function range(start?: number, end?: number, step?: number): Range {
  if (start != null && end == null) {
    end = start
    start = 0
  }
  if (step == null) {
    step = 1
  }
  return { start, end, step }
}

function* enumerateRange(range: Range, size: number): Iterable<number> {
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

function enumerateRanges(ranges: Array<Range>, shape: Shape): Iterable<[number[], number[]]> {
  if (ranges.length !== shape.length) {
    throw new Error('Sizes of ranges and shape are different')
  }
  if (ranges.length === 0) {
    return []
  }

  return ranges.reduce(
    function*(a: Iterable<[number[], number[]]>, r: Range, i): Iterable<[number[], number[]]> {
      for (const [idx1, idx2] of a) {
        let k = 0
        for (const j of enumerateRange(r, shape[i])) {
          yield [idx1.concat([j]), idx2.concat([k])]
          k++
        }
      }
    },
    [[[], []]]
  )
}

function countRange(r: Range, size: number) {
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

  constructor(data: number[], shape: Shape) {
    if (data.length !== shapeProduct(shape)) {
      throw new Error('invalid array and shape')
    }
    this.data = data
    this.shape = shape
    this.size = shapeProduct(shape)
  }

  static empty(shape: number[] | number): NDArray {
    return new NDArray([], [])
  }

  get(indices: number | number[] = 0): number {
    if (typeof indices === 'number') {
      indices = [indices]
    }
    const idx = flattenIndices(indices, this.shape)
    return this.data[idx]
  }

  set(indices: number | number[], value: number) {
    if (typeof indices === 'number') {
      indices = [indices]
    }
    const idx = flattenIndices(indices, this.shape)
    this.data[idx] = value
  }

  update(indices: number | number[], updater: (x: number) => number) {
    if (typeof indices === 'number') {
      indices = [indices]
    }
    const idx = flattenIndices(indices, this.shape)
    this.data[idx] = updater(this.data[idx])
  }

  reshape(shape: number[]): NDArray {
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

  transpose(axes?: number[]): NDArray {
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

  swapAxes(a1: number, a2: number): NDArray {
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

  add(x: Operand, out?: NDArray) {
    return add(this, x, out)
  }
  sub(x: Operand, out?: NDArray) {
    return sub(this, x, out)
  }
  mul(x: Operand, out?: NDArray) {
    return mul(this, x, out)
  }
  div(x: Operand, out?: NDArray) {
    return div(this, x, out)
  }
  pow(x: Operand, out?: NDArray) {
    return pow(this, x, out)
  }
  neg(out?: NDArray) {
    return neg(this, out)
  }
  argMin(axis: number) {
    return argMin(this, axis)
  }
  argMax(axis: number) {
    return argMax(this, axis)
  }

  slice(...indexOrRanges: Array<number | Range | null>): NDArray {
    const indexOrRanges_ = indexOrRanges.filter((e) => e != null) as Array<number | Range>

    if (indexOrRanges_.length > this.shape.length) {
      throw new Error('Too many indices')
    }
    while (indexOrRanges_.length < this.shape.length) {
      indexOrRanges_.push(All)
      indexOrRanges.push(All)
    }

    const ranges = indexOrRanges_.map((ir) => {
      if (typeof ir === 'number') {
        return range(ir, ir + 1)
      }
      return ir
    })
    const resultShape = ranges.map((r, i) => countRange(r, this.shape[i]))
    let result: NDArray

    if (isSameShape(this.shape, resultShape)) {
      result = new NDArray(this.data.slice(), this.shape)
    } else {
      result = zeros(resultShape)
      if (result.size > 0) {
        for (const [idx1, idx2] of enumerateRanges(ranges, this.shape)) {
          result.set(idx2, this.get(idx1))
        }
      }
    }

    const normalizedShape: number[] = []
    let i = 0
    for (const ir of indexOrRanges) {
      if (ir == null) {
        normalizedShape.push(1)
      } else {
        if (typeof ir !== 'number') {
          normalizedShape.push(resultShape[i])
        }
        i++
      }
    }
    return result.reshape(normalizedShape)
  }

  sum(axes?: number | number[]): NDArray {
    return sum(this, axes)
  }

  mean(axes?: number | number[]): NDArray {
    return mean(this, axes)
  }

  clip(min: number, max: number, out?: NDArray): NDArray {
    return clip(this, min, max, out)
  }
}

function isSameShape(shape1: Shape, shape2: Shape): boolean {
  return shape1.length === shape2.length && shape1.every((s1, i) => s1 === shape2[i])
}

export function repeat(x: number, shapeOrNumber: ShapeOrNumber): NDArray {
  const shape = typeof shapeOrNumber === 'number' ? [shapeOrNumber] : shapeOrNumber
  if (!isValidShape(shape)) {
    throw new Error('invalid shape')
  }
  const p = shapeProduct(shape)
  return new NDArray(new Array(p).fill(x), shape)
}

export function zeros(shapeOrNumber: ShapeOrNumber): NDArray {
  return repeat(0, shapeOrNumber)
}

export function zerosLike(array: NDArray): NDArray {
  return zeros(array.shape)
}

function shapeProduct(indices: number[]): number {
  if (indices.length === 0) {
    return 0
  }
  return indices.reduce((a, b) => a * b, 1)
}

function isValidShape(shape: number[]): boolean {
  return shape.every((n) => Number.isFinite(n) && n >= 0)
}

function isReshapable(oldShape: number[], newShape: number[]): boolean {
  return (
    isValidShape(oldShape) &&
    isValidShape(newShape) &&
    shapeProduct(oldShape) === shapeProduct(newShape)
  )
}

function* enumerateIndices(shape: Shape): Iterable<number[]> {
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
      const m = (indices[j] = k % s)
      k = (k - m) / s
    }
    indices[0] = k
    yield indices
  }
}

export function createArray(raw: any[]): NDArray {
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

function flatten(array: any[], dest: any[] = []): any[] {
  for (const a of array) {
    if (Array.isArray(a)) {
      flatten(a, dest)
    } else {
      dest.push(a)
    }
  }
  return dest
}

function flattenIndices(indices: number[], shape: number[]): number {
  const ks = [1]
  let k = 1
  for (let i = shape.length - 1; i >= 1; i--) {
    k *= shape[i]
    ks.unshift(k)
  }
  return indices.reduce((a, idx, i) => a + idx * ks[i], 0)
}

function trim(s: string): string {
  return s.trim()
}
function parseExprForEinsum(expr: string): [Array<Array<string>>, Array<string>] {
  const [a, b] = expr.split('->').map(trim)
  return [
    a.split(';').map((s) => s.split(',').map(trim)),
    b.length > 0 ? b.split(',').map(trim) : []
  ]
}

export function einsum(expr: string, ...arrays: Array<NDArray>): NDArray {
  const [indexNameLists, resultIndexNames] = parseExprForEinsum(expr)
  if (indexNameLists.length === 0) {
    throw new Error('Specify one or more elements for 1st argument')
  }
  for (const [i, indexNames] of indexNameLists.entries()) {
    if (indexNames.length !== arrays[i].shape.length) {
      throw new Error(`Number of index names and rank of array at ${i}`)
    }
  }
  const idByIndexName: { [s: string]: number } = {}
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

function operate(f: BinaryOperator, a: Operand, b: Operand, out?: NDArray): NDArray {
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

  const ma = a.shape.map((s) => (s > 1 ? 1 : 0))
  const mb = b.shape.map((s) => (s > 1 ? 1 : 0))

  const resultShape = a.shape.map((s, i) => Math.max(s, (b as NDArray).shape[i]))
  if (out != null && !isSameShape(out.shape, resultShape)) {
    throw new Error('Shape of `out` is incompatible')
  }

  const result = out || zeros(resultShape)
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

function createUniversalBinaryFunction(f: BinaryOperator): UniversalBinaryOperator {
  return (a: Operand, b: Operand, out?: NDArray) => operate(f, a, b, out)
}

export const add = createUniversalBinaryFunction((a, b) => a + b)
export const sub = createUniversalBinaryFunction((a, b) => a - b)
export const mul = createUniversalBinaryFunction((a, b) => a * b)
export const div = createUniversalBinaryFunction((a, b) => a / b)
export const pow = createUniversalBinaryFunction((a, b) => a ** b)

function operateUnary(f: UnaryOperator, a: Operand, out?: NDArray): NDArray {
  if (typeof a === 'number') {
    a = createArray([a])
  }

  if (out != null && !isSameShape(a.shape, out.shape)) {
    throw new Error('Shape of `out` is incompatible')
  }

  const result = out || zerosLike(a)
  for (const i of enumerateIndices(result.shape)) {
    result.set(i, f(a.get(i)))
  }

  return result
}

function createUniversalUnaryOperator(f: UnaryOperator): UniversalUnaryOperator {
  return (a: Operand, out?: NDArray) => operateUnary(f, a, out)
}

export const neg = createUniversalUnaryOperator((a) => -a)
export const exp = createUniversalUnaryOperator((a) => Math.exp(a))

export function argMin(array: NDArray, axis: number): NDArray {
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

export function argMax(array: NDArray, axis: number): NDArray {
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

function checkAxesForAggregation(
  axesOrAxis: number[] | number | undefined,
  shape: Shape
): [number[], number[]] {
  const dim = shape.length
  if (axesOrAxis == null) {
    return [_range(dim), []]
  }
  let axes: number[]
  if (typeof axesOrAxis === 'number') {
    axes = [axesOrAxis]
  } else {
    axes = axesOrAxis.slice()
  }

  {
    const _axes: number[] = []
    for (const ax of axes) {
      if (ax !== (ax | 0) || ax < 0 || ax >= dim || _axes.includes(ax)) {
        throw new Error('invalid axes')
      }
      _axes.push(ax)
    }
  }

  const remaining = _range(dim).filter((a) => !axes.includes(a))
  return [axes, remaining]
}

export function sum(
  array: NDArray,
  axisOrAxes: number | number[] | undefined = undefined
): NDArray {
  const shape = array.shape
  const [, remainingAxes] = checkAxesForAggregation(axisOrAxes, shape)
  const newShape = remainingAxes.map((a) => shape[a])
  if (newShape.length === 0) {
    let sum = 0
    for (const idx of enumerateIndices(shape)) {
      sum += array.get(idx)
    }
    return createArray([sum])
  }
  const newArray = zeros(newShape)
  for (const idx of enumerateIndices(shape)) {
    const newIdx = remainingAxes.map((a) => idx[a])
    newArray.update(newIdx, (x) => x + array.get(idx))
  }
  return newArray
}

export function mean(
  array: NDArray,
  axisOrAxes: number | number[] | undefined = undefined
): NDArray {
  const s = sum(array, axisOrAxes)
  const sp = shapeProduct(array.shape)
  const newSp = shapeProduct(s.shape)
  if (sp === 0) {
    return s
  }
  return s.div(sp / newSp)
}

export function clip(array: NDArray, min: number, max: number, out?: NDArray): NDArray {
  return operateUnary((n) => Math.max(Math.min(n, max), min), array, out)
}
