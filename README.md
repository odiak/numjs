# NumJS

NumPy-like N-dimensional array implementation for JavaScript. (written in TypeScript)

This project is very in progress, and there are still few APIs. Feature requests and pull requests are welcome.

There are no documentation currently. See spec codes (`*.spec.ts`) or source codes directly.

## Usage

```javascript
import { createArray, einsum } from '@odiak/numjs'
const a = createArray([[1, 2], [3, 4], [5, 6]])
const b = createArray([[1, 2], [3, 4]])

const c = einsum('i,j; j,k -> i,k', a, b)
c.shape // = [2, 2]
c.get([0, 0]) // = 22
c.get([0, 1]) // = 28
c.get([1, 0]) // = 49
c.get([1, 1]) // = 64

const d = a.sum(0) // summing over axis 0
c.shape // = [2]
c.get(0) // = 9
c.get(1) // = 12
```
