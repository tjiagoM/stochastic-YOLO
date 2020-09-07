
for i in range(100):
    with open(f'../TACO/labels/batch_13/{i:06}.txt', 'w') as file:
        # class x_center y_center width height
        file.write('58 0.389578 0.416103 0.038594 0.163146\n')
        file.write('62 0.127641 0.505153 0.233312 0.222700\n')
