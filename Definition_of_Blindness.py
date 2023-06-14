def count_transition(arr):
    count = 0
    transition = False

    for i in range(len(arr)-1):
        if arr[i] == 1 and arr[i+1] == 0:
            transition = True
        elif arr[i] == 0 and arr[i+1] == 1 and transition:
            count += 1
            transition = False

    return count

def transform_array(arr):
    transformed_arr = []

    for num in arr:
        if num <= 0.5:
            transformed_arr.append(0)
        else:
            transformed_arr.append(1)

    return transformed_arr

predictions_l = transform_array(predictions_l)
predictions_r = transform_array(predictions_r)

transition_countL = count_transition(predictions_l)
print("왼쪽눈 전환 횟수:", transition_countL)

transition_countR = count_transition(predictions_r)
print("오른쪽눈 전환 횟수:", transition_countR)
