class stack:
	def __init__(self):
		self.items = []

	def push(self,item):
		self.items.append(item)

	def pop(self):
		return self.items.pop()

	def show_stack(self):
		return self.items

s = stack()
s.push(3)
s.push(4)
for i in range(10):
	s.push(i)
s.pop()
s.pop()
print(s.show_stack())