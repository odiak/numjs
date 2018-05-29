import {
  NDArray,
  zeros,
  createArray,
  einsum,
  add,
  sub,
  mul,
  div,
  pow,
  argMin,
  neg,
  argMax,
  All,
  range
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
      const a = createArray([
        [ [1, 2],
          [3, 4]],
        [ [5, 6],
          [7, 8]]
      ])

      const t = a.transpose([0, 2, 1])
      expect(t.get([0, 0, 1])).to.eq(a.get([0, 1, 0]))
      expect(t.get([0, 1, 0])).to.eq(a.get([0, 0, 1]))
      expect(t.get([1, 0, 1])).to.eq(a.get([1, 1, 0]))
      expect(t.get([1, 1, 0])).to.eq(a.get([1, 0, 1]))
    })
  })

  describe('.swapAxes', () => {
    it('works right', () => {
      const a = createArray([
        [ [1, 2],
          [3, 4]],
        [ [5, 6],
          [7, 8]]
      ])

      const t = a.swapAxes(1, 2)
      expect(t.get([0, 0, 1])).to.eq(a.get([0, 1, 0]))
      expect(t.get([0, 1, 0])).to.eq(a.get([0, 0, 1]))
      expect(t.get([1, 0, 1])).to.eq(a.get([1, 1, 0]))
      expect(t.get([1, 1, 0])).to.eq(a.get([1, 0, 1]))
    })
  })

  describe('.slice', () => {
    it('works right', () => {
      const a = createArray([
        [ [1, 2],
          [3, 4]],
        [ [5, 6],
          [7, 8]]
      ])
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
    })
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
    const a = createArray([
      [1, 2, 3],
      [4, 5, 6]
    ])
    const b = createArray([
      [1],
      [2],
      [3]
    ])
    const c = einsum('i,j; j,k -> i,k', a, b)
    expect(c.shape).to.deep.equal([2, 1])
    expect(c.get([0, 0])).to.equal(14)
    expect(c.get([1, 0])).to.equal(32)
  })

  it('works on case aggregating into scalar', () => {
    const a = createArray([
      [1, 2, 3],
      [4, 5, 6]
    ])
    const n = einsum('i1,i2 ->', a)
    expect(n.shape).to.deep.eq([1])
    expect(n.get()).to.eq(21)
  })
})

describe('add', () => {
  it('works', () => {
    const a = createArray([
      [1, 2, 3],
      [4, 5, 6]
    ])
    const b = createArray([
      [2, 1, 3],
      [3, 4, 0]
    ])
    const c = createArray([
      [1],
      [2]
    ])

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
  })
})

describe('neg', () => {
  it('works right', () => {
    const a = createArray([
      [1, -2],
      [3, 0]
    ])

    const n = neg(a)
    expect(n.get([0, 0])).to.eq(-1)
    expect(n.get([0, 1])).to.eq(2)
    expect(n.get([1, 0])).to.eq(-3)
    expect(n.get([1, 1])).to.eq(0)
  })
})

describe('argMin', () => {
  it('works right for 2d array', () => {
    const a = createArray([
      [2, 1, 3],
      [4, 2, 1]
    ])

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
    const a = createArray([
      [2, 1, 3],
      [4, 2, 1]
    ])

    const m0 = argMax(a, 0)
    expect(m0.get(0)).to.eq(1)
    expect(m0.get(1)).to.eq(1)
    expect(m0.get(2)).to.eq(0)

    const m1 = argMax(a, 1)
    expect(m1.get(0)).to.eq(2)
    expect(m1.get(1)).to.eq(0)
  })
})
