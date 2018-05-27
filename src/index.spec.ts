import { NDArray, zeros, createArray, einsum, add, sub, mul, div, pow } from './index'
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
    const c = einsum([['i', 'j'], ['j', 'k']], ['i', 'k'], a, b)
    expect(c.shape).to.deep.equal([2, 1])
    expect(c.get([0, 0])).to.equal(14)
    expect(c.get([1, 0])).to.equal(32)
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

    const r1 = add(10, 21) as number
    expect(r1).to.eq(31)

    const r2 = add(a, 1) as NDArray
    expect(r2.shape).to.deep.eq([2, 3])
    expect(r2.get([0, 0])).to.eq(2)
    expect(r2.get([0, 1])).to.eq(3)
    expect(r2.get([1, 1])).to.eq(6)

    const r3 = add(a, b) as NDArray
    expect(r3.shape).to.deep.eq([2, 3])
    expect(r3.get([0, 0])).to.eq(3)
    expect(r3.get([0, 1])).to.eq(3)
    expect(r3.get([1, 2])).to.eq(6)

    const r4 = add(a, c) as NDArray
    expect(r4.shape).to.deep.eq([2, 3])
    expect(r4.get([0, 0])).to.eq(2)
    expect(r4.get([1, 1])).to.eq(7)
  })
})
