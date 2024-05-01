def bootstrapping(data, num):
    sample = np.random.choice(data, num, replace=True)
    return sample

def data_loader(data, iters, num):
    dataset = []
    for _ in range(iters):
        sample = bootstrapping(data, num)
        dataset.append(sample)
    return np.array(dataset)

num = 60
iters = 10000

exponential_dataset = data_loader(data_exp, iters, num)
hyperbolic_dataset = data_loader(data_hyp, iters, num)

print(exponential_dataset.shape)
>>> (10000, 60)

print(hyperbolic_dataset.shape)
>>> (10000, 60)