# Basic syntax: array[start:stop:step]
my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print("Original list:", my_list)

# Simple slicing
list1 = my_list[2:5]  # [2, 3, 4]
print("list1 (2:5):", list1)
list2 = my_list[:5]  # [0, 1, 2, 3, 4]     # start defaults to 0
print("list2 (:5):", list2)
list3 = my_list[5:]  # [5, 6, 7, 8, 9]     # end defaults to len(list)
print("list3 (5:):", list3)
list4 = my_list[:]  # [0,...,9]           # creates a shallow copy
print("list4 (:):", list4)

# Using step
list5 = my_list[::2]  # [0, 2, 4, 6, 8]     # every second element
print("list5 (::2):", list5)
list6 = my_list[1::2]  # [1, 3, 5, 7, 9]     # starting from index 1
print("list6 (1::2):", list6)

# Negative indices
list7 = my_list[-3:]  # [7, 8, 9]           # last three elements
print("list7 (-3:):", list7)
list8 = my_list[:-2]  # [0,...,7]           # everything except last two
print("list8 (:-2):", list8)
list9 = my_list[::-1]  # [9,8,7,6,5,4,3,2,1,0] # reverse the list
print("list9 (::-1):", list9)
