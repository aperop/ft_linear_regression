from LinReg import LinReg


def train() -> None:
    model = LinReg('data.csv')
    weight = model.fit()
    print(f'\ntheta0: {weight[0]:.4}\ntheta1: {weight[1]:.5}')
    print(f'R2_score: {model.score():.5}')
    model.save()
    model.plot().show()
    model.plot_cost().show()


if __name__ == '__main__':
    train()
