from sklearn.metrics import confusion_matrix
y_pred = ['a','a','b','b','a','b','a','a','a','a']
y_true  = ['a','b','b','a','b','a','a','b','a','b']
cm = confusion_matrix(y_true, y_pred)
print(cm)
#=======================================================================

y_pred = ['a','b','c','a','b','c','a','b','c','a']
y_true =  ['a','a','b','b','a','b','c','c','b','b']
cm = confusion_matrix(y_true, y_pred)
print(cm)

#=======================================================================


y_pred = [5,8,9,9,8,5,5,9,8,5,9,8]
y_true =  [9,9,8,8,5,5,9,5,8,9,8,5]
cm = confusion_matrix(y_true, y_pred)
print(cm)
