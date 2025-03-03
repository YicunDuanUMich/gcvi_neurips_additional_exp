import sbibm

if __name__ == "__main__":
    task = sbibm.get_task("bernoulli_glm")
    prior = task.get_prior()
    simulator = task.get_simulator()
    
    thetas = prior(num_samples=16)
    xs = simulator(thetas)
