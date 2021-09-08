import random
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


class Population:

    # def __init__(self, size, lower_bound, upper_bound, a, b, c, precision, pc, pm, stages ):
    def __init__(self, file_name):

        def generate_initial_population():  # construieste cei "size" cromozomi
            initial_population = []
            for i in range(self.size):
                chromosome = []
                for j in range(self.L):
                    chromosome.append(random.randint(0, 1))
                initial_population.append(chromosome)

            return initial_population

        f = open(file_name, "r")    # citire din fisier
        self.size = int(f.readline().split()[0])
        self.lower_bound, self.upper_bound = map(int, f.readline().split())
        self.a, self.b, self.c = map(float, f.readline().split())
        self.precision = int(f.readline().split()[0])
        self.pc, self.pm = map(float, f.readline().split())
        self.stages = int(f.readline().split()[0])
        self.L = int(math.log((self.upper_bound - self.lower_bound) * 10 ** self.precision, 2)) + 1
        self.chromosomes = generate_initial_population()

    def transformation(self, chromosome):   # transformarea liniara
        x10 = 0
        for i in range(self.L):
            x10 += 2 ** (self.L - i - 1) * chromosome[i]
        return (self.upper_bound - self.lower_bound) * x10 / (2 ** self.L - 1) + self.lower_bound

    def fitness(self, chromosome):  # functia de fitness
        x = self.transformation(chromosome)
        # return self.a * x ** 2 + self.b * x + self.c
        return self.g(x)

    def f(self, x):
        return self.a * x ** 2 + self.b * x + self.c

    def g(self, x):
        return x ** 3 + 3 * x ** 2 - 4 * x + 7

    def print_population(self):     # informatiile despre populatia curenta
        print("-- Population --\n")     # cromozomii cu valoarea din b12 si fitness-ul
        for i in range(self.size):
            x = self.chromosomes[i]
            print(i + 1, ": ", *x, sep="")
            print("  x= ", self.transformation(x), " f= ", self.fitness(x))

    def total_performance(self):    # suma functiilor fitness
        tp = 0
        for x in self.chromosomes:
            tp += self.fitness(x)
        return tp

    def elitist_member(self):   # returneaza membrul cu functia fitness maxima
        return max(self.chromosomes, key=self.fitness)

    # calculeaza probabilitatile pentru selectie a cromozomilor
    def selection_probabilities(self):  # metoda ruletei
        probabilities = [0 for _ in range(self.size)]
        tp = self.total_performance()
        for i in range(self.size):
            x = self.chromosomes[i]
            probabilities[i] = (self.fitness(x) / tp)
        return probabilities

    def probability_intervals(self):    # genereaza intervalele probabilitatilor de selectie
        intervals = [0]
        probabilities = self.selection_probabilities()
        for i in range(self.size):
            intervals.append(intervals[i] + probabilities[i])
        return intervals

    def print_probabilities(self):  # afisare informatii probabilitati
        print("\nSelection probabilities\n")
        probabilities = self.selection_probabilities()
        for i in range(self.size):
            print("Chromosome ", i + 1, ": p= ", probabilities[i])
        intervals = self.probability_intervals()
        print("\nProbability intervals: \n")
        print(intervals)

    # selectia cromozomilor
    def proportional_selection(self, elitist_criteria=True, printable=False):

        def binary_search(arr, start, end, x):
            if end >= start:
                mid = start + (end - start) // 2
                if arr[mid] <= x <= arr[mid + 1]:
                    return mid
                elif arr[mid] >= x:
                    return binary_search(arr, start, mid - 1, x)
                else:
                    return binary_search(arr, mid + 1, end, x)
            else:
                return -1

        new_chromosomes = []    # noua populatie
        intervals = self.probability_intervals()
        if elitist_criteria is True:
            size = self.size - 1
            if printable:
                print("\nMembrul elitist: ", *self.elitist_member(), sep="")
                print()
            new_chromosomes.append(self.elitist_member())
        else:
            size = self.size
        for i in range(size):
            u = random.uniform(0, 1)
            index = binary_search(intervals, 0, self.size, u)
            if printable:
                # print("Intervale probabilitati selectie")
                print("u= ", u, " cromozomul ", index + 1)
            new_chromosomes.append(self.chromosomes[index])
        self.chromosomes = new_chromosomes

    def crossover_selected(self, printable=False):  # selectie pentru crossover
        selected_chromosomes = []                   # tinand cont de probabilitatea pc
        for i in range(self.size):
            u = random.uniform(0, 1)
            if printable:
                print(i + 1, ": ", *self.chromosomes[i], sep="")
                print("  u: ", u)
            if u <= self.pc:
                if printable:
                    print("participa")
                selected_chromosomes.append(i)
        return selected_chromosomes

    def population_crossover(self, printable=False):

        def crossover2(ch1, ch2):
            u = random.randint(2, self.L)   # punct de rupere
            if printable:
                print("punct= ", u)
                ch1[:self.L-u], ch2[u:] = ch2[u:], ch1[:self.L-u]

        def crossover3(ch1, ch2, ch3):  # in cazul in care am selectat un nr impar de cromozomi
            u = random.randint(0, self.L)
            if printable:
                print("punct= ", u)
            if u != 0:
                ch1[:u], ch2[:u], ch3[:u] = ch3[:u], ch1[:u], ch2[:u]
                # ch1[:self.L - u], ch2[u:], ch3[:self.L - u] = ch2[u:], ch3[:self.L - u], ch1[u:]

        selected_chromosomes = self.crossover_selected()
        if len(selected_chromosomes) % 2 == 1 and len(selected_chromosomes) >= 3:
            i1 = random.randint(0, len(selected_chromosomes) - 1)
            ic1 = selected_chromosomes[i1]
            selected_chromosomes.pop(i1)
            i2 = random.randint(0, len(selected_chromosomes) - 1)
            ic2 = selected_chromosomes[i2]
            selected_chromosomes.pop(i2)
            if len(selected_chromosomes) == 1:
                i3 = 0
            else:
                i3 = random.randint(0, len(selected_chromosomes) - 1)
            ic3 = selected_chromosomes[i3]
            selected_chromosomes.pop(i3)
            if printable:
                print("Cromozomii ", ic1+1, ic2+1, ic3+1)
                print(*self.chromosomes[ic1], sep="")
                print(*self.chromosomes[ic2], sep="")
                print(*self.chromosomes[ic3], sep="")
            crossover3(self.chromosomes[ic1], self.chromosomes[ic2],
                       self.chromosomes[ic3])
            if printable:
                print("Rezultat")
                print(*self.chromosomes[ic1], sep="")
                print(*self.chromosomes[ic2], sep="")
                print(*self.chromosomes[ic3], sep="")

        while len(selected_chromosomes) > 1:
            i1 = random.randint(0, len(selected_chromosomes) - 1)
            ic1 = selected_chromosomes[i1]
            selected_chromosomes.pop(i1)
            if len(selected_chromosomes) == 1:
                i2 = 0
            else:
                i2 = random.randint(0, len(selected_chromosomes) - 1)
            ic2 = selected_chromosomes[i2]
            selected_chromosomes.pop(i2)
            if printable:
                print("Cromozomii ", ic1+1, ic2+1)
                print(*self.chromosomes[ic1], sep="")
                print(*self.chromosomes[ic2], sep="")
            crossover2(self.chromosomes[ic1], self.chromosomes[ic2])
            if printable:
                print("Rezultat")
                print(*self.chromosomes[ic1], sep="")
                print(*self.chromosomes[ic2], sep="")

    # mutatia rara si mutatia regulara
    def rare_mutation(self, printable=False):
        modified = []
        for i in range(self.size):
            u = random.uniform(0, 1)
            if u < self.pm:
                modified.append(i + 1)
                index = random.randint(0, self.L - 1)
                self.chromosomes[i][index] = 1 - self.chromosomes[i][index]
        if printable:
            print("Cromozomi modificati: ", *modified, sep="")

    def regular_mutation(self, printable=False):
        modified = set()
        for i in range(self.size):
            for j in range(self.L):
                u = random.uniform(0, 1)
                if u < self.pm:
                    modified.add(i + 1)
                    self.chromosomes[i][j] = 1 - self.chromosomes[i][j]
        if printable:
            print("Cromozomi modificati: ", *modified, sep=" ")

    def population_mutation(self, mutation="Regular mutation", printable=False):
        if mutation == "Regular mutation":
            self.regular_mutation(printable=printable)
        elif mutation == "Rare mutation":
            self.rare_mutation(printable=printable)

    # genereaza graficul functiei fitness, cat si punctul generat de algoritm
    def show_graph(self):
        x = np.arange(self.lower_bound, self.upper_bound, 0.1)
        y = self.g(x)
        elitist_member = self.elitist_member()
        x1 = self.transformation(elitist_member)
        y1 = self.fitness(elitist_member)
        plt.plot(x, y, label='functia f')
        plt.plot(x1, y1, color='green', linestyle='dashed', linewidth=3,
                 marker='o', markerfacecolor='blue', markersize=12)
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.title('f function and our result!')
        plt.show()

    def main(self):
        print("Populatie initiala")
        self.print_population()
        self.print_probabilities()
        print("\nSelection:\n")
        self.proportional_selection(printable=True)
        self.print_population()
        print("\nCrossover\n")
        self.population_crossover(printable=True)
        print("\nMutation:\n")
        self.population_mutation("Regular mutation", printable=True)
        self.print_population()
        print("\nElitist member's evolution")
        elitist = self.elitist_member()
        fr = 1
        for i in range(1, self.stages):
            self.proportional_selection()
            self.population_crossover()
            self.population_mutation("Regular mutation")
            el = self.elitist_member()
            print(i + 1, ": ", self.transformation(el), self.fitness(el))
            # if elitist == el:
            #     fr += 1
            # else:
            #     fr = 0
            #
            # if fr == 10:
            #     print("\nNSTOP dupa: ", i, " pasi")
            #     break
        self.show_graph()


p = Population("population.in")
p.main()
