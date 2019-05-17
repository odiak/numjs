import {
  NDArray,
  repeat,
  zeros,
  createArray,
  einsum,
  add,
  argMin,
  neg,
  argMax,
  All,
  NewAxis,
  range,
  sum,
  mean,
  clip,
  min,
  max,
  abs
} from './index'
import { expect } from 'chai'
import 'mocha'

describe('NDArray', () => {
  it('should work correctly', () => {
    const a = new NDArray([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])
    // [[[1,2], [3,4]], [[5, 6], [7, 8]]]

    expect(a.get([1, 1, 0])).to.equal(7)
    a.set([1, 1, 0], -1)
    expect(a.get([1, 1, 0])).to.equal(-1)

    const b = new NDArray([1, 2, 3, 4, 5, 6], [2, 3])
    expect(b.get([0, 0])).to.equal(1)
    expect(b.get([0, 1])).to.equal(2)
    expect(b.get([0, 2])).to.equal(3)
    expect(b.get([1, 0])).to.equal(4)
    expect(b.get([1, 1])).to.equal(5)
    expect(b.get([1, 2])).to.equal(6)
  })

  describe('.size', () => {
    it('returns correct value', () => {
      const a = zeros([2, 3, 4, 5])
      expect(a.size).to.eq(120)
    })
  })

  describe('.reshape', () => {
    it('works', () => {
      const a = new NDArray([1, 2, 3, 4, 5, 6], [2, 3])
      const b = a.reshape([3, 2])
      expect(b.get([1, 1])).to.equal(4)

      expect(a.reshape([-1, 2]).shape).to.deep.equal([3, 2])
    })

    it('fails', () => {
      const a = new NDArray([1, 2, 3, 4], [2, 2])
      expect(() => {
        a.reshape([3, 4])
      }).to.throw()

      expect(() => {
        a.reshape([-1, -1])
      }).to.throw()

      expect(() => {
        a.reshape([5, -1])
      }).to.throw()
    })
  })

  describe('.transpose', () => {
    it('works', () => {
      const a = new NDArray([1, 2, 3, 4, 5, 6], [3, 2])
      const b = a.transpose()
      expect(a.get([0, 1])).to.equal(b.get([1, 0]))
      expect(a.get([2, 0])).to.equal(b.get([0, 2]))
    })

    it('works right with axes', () => {
      const a = createArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

      const t = a.transpose([0, 2, 1])
      expect(t.get([0, 0, 1])).to.eq(a.get([0, 1, 0]))
      expect(t.get([0, 1, 0])).to.eq(a.get([0, 0, 1]))
      expect(t.get([1, 0, 1])).to.eq(a.get([1, 1, 0]))
      expect(t.get([1, 1, 0])).to.eq(a.get([1, 0, 1]))
    })
  })

  describe('.swapAxes', () => {
    it('works right', () => {
      const a = createArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

      const t = a.swapAxes(1, 2)
      expect(t.get([0, 0, 1])).to.eq(a.get([0, 1, 0]))
      expect(t.get([0, 1, 0])).to.eq(a.get([0, 0, 1]))
      expect(t.get([1, 0, 1])).to.eq(a.get([1, 1, 0]))
      expect(t.get([1, 1, 0])).to.eq(a.get([1, 0, 1]))
    })
  })

  describe('.slice', () => {
    it('works right', () => {
      const a = createArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      const s1 = a.slice()
      expect(s1.shape).to.deep.eq([2, 2, 2])
      expect(s1.data).to.deep.eq([1, 2, 3, 4, 5, 6, 7, 8])

      const s2 = a.slice(1, All, All)
      expect(s2.shape).to.deep.eq([2, 2])
      expect(s2.data).to.deep.eq([5, 6, 7, 8])

      const s3 = a.slice(All, 0, All)
      expect(s3.shape).to.deep.eq([2, 2])
      expect(s3.data).to.deep.eq([1, 2, 5, 6])

      const b = createArray([
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 0, 1, 2],
        [3, 4, 5, 6, 7, 8],
        [9, 0, 1, 2, 3, 4]
      ])

      const s4 = b.slice(range(1, 3), range(1, 4, 2))
      expect(s4.shape).to.deep.eq([2, 2])
      expect(s4.data).to.deep.eq([8, 0, 4, 6])

      const s5 = b.slice(All, NewAxis, NewAxis, All) // add new axis
      expect(s5.shape).to.deep.eq([4, 1, 1, 6])
      expect(s5.data).to.deep.eq(b.data)

      const s6 = b.slice(All, NewAxis, 1)
      expect(s6.shape).to.deep.eq([4, 1])
      expect(s6.data).to.deep.eq([2, 8, 4, 0])
    })
  })
})

describe('repeat', () => {
  it('works', () => {
    const a = repeat(2, [3, 3])
    expect(a.get([1, 0])).to.equal(2)
    expect(a.get([2, 2])).to.equal(2)
  })

  it('fails', () => {
    expect(() => {
      zeros([-1, -1])
    }).to.throw()
  })
})

describe('zeros', () => {
  it('works', () => {
    const a = zeros([3, 3])
    expect(a.get([1, 0])).to.equal(0)
    expect(a.get([2, 2])).to.equal(0)
  })

  it('fails', () => {
    expect(() => {
      zeros([-1, -1])
    }).to.throw()
  })
})

describe('createArray', () => {
  it('works', () => {
    const a = createArray([[1, 2, 3], [4, 5, 6]])
    expect(a.shape).to.deep.equal([2, 3])
    expect(a.get([1, 1])).to.equal(5)
  })

  it('fails', () => {
    expect(() => {
      createArray([[1, 2, 3], [4, 5, 6, 7]])
    }).to.throw()
  })
})

describe('einsum', () => {
  it('works', () => {
    const a = createArray([[1, 2, 3], [4, 5, 6]])
    const b = createArray([[1], [2], [3]])
    const c = einsum('i,j; j,k -> i,k', a, b)
    expect(c.shape).to.deep.equal([2, 1])
    expect(c.get([0, 0])).to.equal(14)
    expect(c.get([1, 0])).to.equal(32)
  })

  it('works on case aggregating into scalar', () => {
    const a = createArray([[1, 2, 3], [4, 5, 6]])
    const n = einsum('i1,i2 ->', a)
    expect(n.shape).to.deep.eq([1])
    expect(n.get()).to.eq(21)
  })
})

describe('add', () => {
  it('works', () => {
    const a = createArray([[1, 2, 3], [4, 5, 6]])
    const b = createArray([[2, 1, 3], [3, 4, 0]])
    const c = createArray([[1], [2]])

    const r1 = add(10, 21).get()
    expect(r1).to.eq(31)

    const r2 = add(a, 1)
    expect(r2.shape).to.deep.eq([2, 3])
    expect(r2.get([0, 0])).to.eq(2)
    expect(r2.get([0, 1])).to.eq(3)
    expect(r2.get([1, 1])).to.eq(6)

    const r3 = add(a, b)
    expect(r3.shape).to.deep.eq([2, 3])
    expect(r3.get([0, 0])).to.eq(3)
    expect(r3.get([0, 1])).to.eq(3)
    expect(r3.get([1, 2])).to.eq(6)

    const r4 = add(a, c)
    expect(r4.shape).to.deep.eq([2, 3])
    expect(r4.get([0, 0])).to.eq(2)
    expect(r4.get([1, 1])).to.eq(7)

    const d = zeros([2, 3])
    const r5 = add(a, c, d) // specifying out array
    expect(r5).to.eq(d)
    expect(r5.data).to.deep.eq(r4.data)
  })
})

describe('neg', () => {
  it('works right', () => {
    const a = createArray([[1, -2], [3, 0]])

    const n = neg(a)
    expect(n.get([0, 0])).to.eq(-1)
    expect(n.get([0, 1])).to.eq(2)
    expect(n.get([1, 0])).to.eq(-3)
    expect(n.get([1, 1])).to.eq(0)

    const b = zeros(a.shape)
    const m = neg(a, b) // specifying out array
    expect(m).to.eq(b)
    expect(m.data).to.deep.eq(n.data)
  })
})

describe('argMin', () => {
  it('works right for 2d array', () => {
    const a = createArray([[2, 1, 3], [4, 2, 1]])

    const m0 = argMin(a, 0)
    expect(m0.get(0)).to.eq(0)
    expect(m0.get(1)).to.eq(0)
    expect(m0.get(2)).to.eq(1)

    const m1 = argMin(a, 1)
    expect(m1.get(0)).to.eq(1)
    expect(m1.get(1)).to.eq(2)
  })
})

describe('argMax', () => {
  it('works right for 2d array', () => {
    const a = createArray([[2, 1, 3], [4, 2, 1]])

    const m0 = argMax(a, 0)
    expect(m0.get(0)).to.eq(1)
    expect(m0.get(1)).to.eq(1)
    expect(m0.get(2)).to.eq(0)

    const m1 = argMax(a, 1)
    expect(m1.get(0)).to.eq(2)
    expect(m1.get(1)).to.eq(0)
  })
})

describe('sum', () => {
  it('works right for 1d array', () => {
    const a = createArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    const sum1 = sum(a)
    expect(sum1.get()).to.eq(55)
    expect(sum1.shape).to.deep.eq([1])

    const sum2 = sum(a, [0])
    expect(sum2.get()).to.eq(55)
    expect(sum2.shape).to.deep.eq([1])
  })

  it('works right for 3d array', () => {
    const a = createArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

    const sum1 = sum(a, [1, 2])
    expect(sum1.shape).to.deep.eq([3])
    expect(sum1.get(0)).to.eq(10)
    expect(sum1.get(1)).to.eq(26)
    expect(sum1.get(2)).to.eq(42)

    const sum2 = sum(a, [0])
    expect(sum2.shape).to.deep.eq([2, 2])
    expect(sum2.get([0, 0])).to.eq(15)
    expect(sum2.get([1, 0])).to.eq(21)

    const sum3 = sum(a, [])
    expect(sum3.shape).to.deep.eq(a.shape)
    expect(sum3.data).to.deep.eq(a.data)

    const sum4 = sum(a)
    expect(sum4.shape).to.deep.eq([1])
    expect(sum4.get()).to.eq(78)
  })
})

describe('mean', () => {
  it('works right for 1d array', () => {
    const a = createArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    const m1 = mean(a)
    expect(m1.shape).to.deep.eq([1])
    expect(m1.get()).to.eq(5.5)

    const m2 = mean(a, [0])
    expect(m2.shape).to.deep.eq([1])
    expect(m2.get()).to.eq(5.5)
  })

  it('works right for 3d array', () => {
    const a = createArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])

    const m1 = mean(a, [1, 2])
    expect(m1.shape).to.deep.eq([3])
    expect(m1.get(0)).to.eq(2.5)
    expect(m1.get(1)).to.eq(6.5)
    expect(m1.get(2)).to.eq(10.5)

    const m2 = mean(a, [])
    expect(m2.shape).to.deep.eq(a.shape)
    expect(m2.data).to.deep.eq(a.data)

    const m3 = mean(a)
    expect(m3.shape).to.deep.eq([1])
    expect(m3.get()).to.eq(6.5)
  })
})

describe('clip', () => {
  it('works', () => {
    const a = createArray([[1, 2, 3], [5, 3, 10]])

    const c = clip(a, 2, 5)
    expect(c.shape).to.deep.eq(a.shape)
    expect(c.data).to.deep.eq([2, 2, 3, 5, 3, 5])

    const d = zeros(a.shape)
    const e = clip(a, 2, 5, d)
    expect(e).to.eq(d)
    expect(e.data).to.deep.eq(c.data)
  })
})

describe('min', () => {
  it('works', () => {
    const a = createArray([[3, 2, 4], [5, 1, -1]])

    const m1 = min(a, 1)
    expect(m1.get(0)).to.eq(2)
    expect(m1.get(1)).to.eq(-1)

    const m2 = min(a)
    expect(m2.get()).to.eq(-1)
  })
})

describe('max', () => {
  it('works', () => {
    const a = createArray([[3, 2, 4], [5, 1, -1]])

    const m1 = max(a, 1)
    expect(m1.get(0)).to.eq(4)
    expect(m1.get(1)).to.eq(5)

    const m2 = max(a)
    expect(m2.get()).to.eq(5)
  })
})

describe('abs', () => {
  it('works', () => {
    const a = createArray([0, -2, 3, 1])

    const b = abs(a)
    expect(b.get(0)).to.eq(0)
    expect(b.get(1)).to.eq(2)
    expect(b.get(2)).to.eq(3)
    expect(b.get(3)).to.eq(1)
  })
})
