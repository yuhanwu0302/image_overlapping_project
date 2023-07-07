a = [6,1,3,2,3,1,3,4]
import math
def sliding(s,arr):
    output = math.inf
    curr_sum = 0
    curr_length = 0
    start =0
    for end in range(len(arr)):
        curr_sum += arr[end]
        curr_length += 1
        while curr_sum >= s:
            output = min(output,curr_length)
            curr_sum -= arr[start]
            start += 1
            curr_length -= 1
    if output == math.inf:
        return 0
    return output
sliding(23,a) 

