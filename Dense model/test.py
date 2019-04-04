fred = 18
barney = FRED = 44;                     # Case sensistive.
bill = (fred + barney * FRED - 10)
alice = 10 + bill / 100                 # Regular division does not truncate.
alice2 = 10 + bill // 100               # But the // operator provides int div.
frank = 10 + float(bill) / 100
print("fred =", fred)
print("FRED =", FRED)
print("bill =", bill)
print("alice =", alice)
print("alice2 =", alice2)
print("frank =", frank )
print()
