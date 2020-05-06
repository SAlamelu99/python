list1=[12,-7,5,64,-14]
list2=[12,14,-95,3]
i=0
list3=[]

for i in list1:
    if i>=0:
        print(i,end=" ")
    
for i in range(0,len(list2)):
    if list2[i]>0:
        list3.append(list2[i])  
    i=i+1
print(list3)
