def readFilePairs(filepath):
    times_done = False
    times = []
    machines = []

    with open(filepath) as fp:
        line = fp.readline()
        n, mn = line.strip().split(' ')
        line = fp.readline()

        while line:
            parse_line = ' '.join(line.split())
            raw_line = parse_line.strip().split(' ')
            curr = []
            i = 0
            machine = []
            time = []
            while i < len(raw_line):
                m, t = raw_line[i], raw_line[i + 1]
                machine.append(int(m))
                time.append(int(t))
                i += 2

            times.append(time)
            machines.append(machine)
            line = fp.readline()

    return times, machines, int(n)
    