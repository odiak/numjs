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
  })

  describe('.reshape', () => {
    it('works', () => {
      const a = new NDArray([1, 2, 3, 4, 5, 6], [2, 3])
      const b = a.reshape([3, 2])
      expect(b.get([1, 1])).to.equal(5)
    })

    it('fails', () => {
      const a = new NDArray([1, 2, 3, 4], [2, 2])
      expect(() => {
        a.reshape([3, 4])
      }).to.throw()
    })
  })
})
