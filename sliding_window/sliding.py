def slidingwindow(k,arr):
    start = 0
    sum = 0.0
    result = []
    for end in range((len(arr))):
        sum += arr[end]
        if end >= k-1:
            result.append(sum/k)
            sum -= arr[start]
            start += 1
    return result 

