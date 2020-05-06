#assigning elements to differnt list
list1=[1,2,3,4,5]
list2=[9,8,7,6,5]
print("before assigning list")
print("list1: ",list1)
print("list2: ",list2)
list1.append(6)
list2.append(4)
list1.insert(6,7)
list2.insert(6,"three")
print("after assigning list1: ",list1)
print("after assigning list2: ",list2)
#accessing elements from a Tuple
tuple=(1,2,"three","a")
print("tuple:",tuple)
print("accessing the element'three'")
print(tuple[2])
print("accessing the whole Tuple")
print(tuple[0:len(tuple)])
#deleting different dictionarfy elements
dic={"purushoth":1713103,"alamelu":1713104,"charumathi":1713105}
print("before deleting: ", dic)
del dic["purushoth"]
print("after deleting: ",dic)
