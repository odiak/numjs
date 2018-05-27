export function range (start: number, end?: number, step?: number): number[] {
  if (end == null) {
    end = start
    start = 0
  }
  if (step == null) {
    step = 1
  }

  if (!Number.isFinite(start)) {
    throw new Error('Invalid start')
  }
  if (!Number.isFinite(end)) {
    throw new Error('Invalid end')
  }
  if (step === 0 || !Number.isFinite(step)) {
    throw new Error('Invalid step')
  }

  const result = []
  if (step > 0) {
    for (let i = start; i < end; i += step) {
      result.push(i)
    }
  } else {
    for (let i = start; i > end; i += step) {
      result.push(i)
    }
  }
  return result
}
