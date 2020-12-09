txt = "The best things in life are free!"
print(len(txt.split()))
a = txt.split()
print(a)
for i in range(0,len(a)):
    if i ==2:
        a.remove(a[i])
print(a)

txt.replace(a[2],'')
print(txt.replace(a[2],''))
str1 = ' '
for s in a:
    str1 = str1 + ' ' + s
print(str1)