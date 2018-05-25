import { NDArray } from './index'
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