# x1 = np.tile(np.arange(100), (4, 2)).T
# figure(1)
# plot(x1, activation=True, too_many_col_check=False)
#
# df1 = pd.DataFrame(data=np.random.randn(10, 3))
# df1.columns = ["abc", "def", "xyz"]
# df1.set_index(pd.date_range("2017-08-30", periods=len(df1)), inplace=True)
# figure(2)
# plot(df1, line=[True, False, True], id=[1, 2, 3])

df2 = pd.DataFrame(data=np.random.randn(5, 3), index=["h1", "h2", "h3", "h4", "h5"])
df2.columns = ["abc", "def", "xyz"]
df2["abc"] = ["III", "JJJ", "KKk", "hoo", "gee"]


n = 5
x1, y1 = np.meshgrid(np.arange(n), np.arange(n))
x1 = x1.ravel()
y1 = y1.ravel()
# strength = np.random.randint(0, 100, len(x1))
strength = np.zeros((n, n))
for i, j in itertools.product(range(n), range(n)):
    strength[i, j] = i - j
strength = strength.ravel()

bubble = np.zeros((n, n))
for i, j in itertools.product(range(n), range(n)):
    # bubble[i, j] = i * j
    bubble[i, j] = np.random.randn()
bubble = bubble.ravel()

plot(np.stack((x1, y1, bubble, strength)).T, single_series=True)

x = np.random.randn(5, 4)
plot(x, assignment={"mainX": 0, "mainY": 1, "bubbleRadius": 2}, style="color-bubble")

xpd = pd.DataFrame(x, columns=["aa", "bb", "cc", "dd"])
xpd["dd"] = ["good", "better", "bad", "great", "worst"]
plot(xpd, color=10, assignment={"mainX": "index", "mainY": "bb", "bubbleRadius": 2, "colorStrength": "dd"}, style="color-bubble")


# plot(x1, np.stack((x1, y1, strength)).T)
# plot(np., is_single_series=True)
