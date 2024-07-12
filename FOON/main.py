with open('FOON.txt') as f:
    lines = f.readlines()
new_file = open('motion.txt', 'w')

motions = []

for line in lines:
    if line.startswith('M'):
        motion = line.split('\t')[1]
        if motion in motions:
            continue
        else:
            motions.append(motion)
            new_file.write(motion.rstrip() + '\t' + '0.1' + '\n')
new_file.close()