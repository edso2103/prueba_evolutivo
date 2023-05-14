import matplotlib.pyplot as plt
import numpy as np
import random
import turtle
from scipy.stats import rankdata
from PIL import Image, ImageDraw
from skimage.transform import resize
import io
import os

class Poblacion:
    
    def __init__(self, num_individuos,n_genes):
        self.num_individuos = num_individuos
        self.individuos = np.array([])
        self.fitness = np.array([])
        self.n_genes = n_genes

    def inicializarPoblacion(self):
        for i in range(0,self.num_individuos):
            individuo = Individuo()
            individuo.crearIndividuo(self.n_genes)
            self.individuos = np.append(self.individuos, individuo)        
    
    def calcularFitness(self):
        self.fitness = np.array([])
        for individuo in self.individuos:
            fitness = individuo.actualizarFitness()
            self.fitness = np.append(self.fitness, fitness)

#["seleccionRuleta","seleccionEstocastica","seleccionRankin
        return self.fitness

    def reemplazarGeneracion(self, hijos):
        self.individuos = np.array(hijos)
        self.fitness = np.array([])
        return self.individuos
    
    def getMejorIndividuo(self):
        mejor_fitness = np.min(self.fitness)
        index = np.where(self.fitness == mejor_fitness)[0][0]
        return self.individuos[index]
        
    def __str__(self):
        return "\n".join(str(individuo) for individuo in self.individuos)

class Individuo:
    
    def __init__(self):
        self.genes = np.array([], dtype=np.int8)
        self.fitness = None

    def verificarIndividuo(self, cadena):
        num_abiertos = cadena.count("[")  
        num_cerrados = cadena.count("]")  

        if num_abiertos != num_cerrados:
            return False

        # verificamos que los corchetes estén balanceados
        pila = []
        for c in cadena:
            if c == '[':
                pila.append('[')
            elif c == ']':
                if len(pila) == 0 or pila.pop() != '[':
                    return False

        # verificamos que los corchetes estén correctamente anidados
        pila = []
        for c in cadena:
            if c == '[':
                pila.append('[')
            elif c == ']':
                if len(pila) == 0 or pila.pop() != '[':
                    return False
                elif len(pila) == 0:
                    return True
        return False

    def crearIndividuo(self, n_genes):
        lista = ['F', '+', '-', '[', ']']
        probabilidades = [0.5, 0.15, 0.15, 0.1, 0.1]

        while True:
            genes = random.choices(lista, weights=probabilidades, k=n_genes)
            f = ''.join(genes)
            if self.verificarIndividuo(f):
                self.genes = np.array(genes)
                break


    def actualizarFitness(self):
            
            '''
            La idea que se plantea es generar diferentes arboles que tenga cierta similud entre sí para ello se tienen en cuenta las siguientes reglas:
            * FFF para obtener un tronco largo
            * Se usaran dos conjuntos de corchetes los cuales pueden tener 5 caracteres ya sea +/- y al menos una F.
            * Los signos dentro de los corchetes deben ser iguales para evitar que se forme una espiral.
            * El primer signo antes del "[" debe ser igual a los siguientes dentro de "[" "]", al igual que con el otro conjunto de corchetes
            * El signo antes del segundo corchete abierto debe ser diferente al signo antes del primer corchete abierto

            '''
            score = 0  

            a  = np.array(["F"])
            a_ = np.array([0,1,2])

            b  = np.array(["+","-"])
            b_ = np.array([3,11])
            
            c  = np.array(["["])
            c_ = np.array([4,12])

            d  = np.array(["]"])
            d_ = np.array([10,18])

            e  = np.array(["+","F"])
            e_ = np.array([5,6,7,8,9])

            f =  np.array(["-","F"])
            f_ = np.array([13,14,15,16,17])

            validar_aux = np.array([])
            elemento_primer_corchete = np.array([])
            elemento_segundo_corchete = np.array([])
            for i in range(len(self.genes)):

                if(any(np.isin(a_, i))):
                    if(any(np.isin(a, self.genes[i]))):
                        score += 1
                    else:
                        score -= 1

                if(any(np.isin(b_, i))):
                    if(len(validar_aux)==0):
                        if(any(np.isin(b, self.genes[i]))):
                            score += 1
                            validar_aux = [i,self.genes[i]]
                        else:
                            score -= 1  
                        
                    else:                
                        if(validar_aux[0]==3):
                                if(validar_aux[1]=="+"):
                                    if(i==11):
                                        if(self.genes[i]=="-"):
                                            score += 1
                                        else:
                                            score -= 1
                                else:
                                    if(i==11):
                                        if(self.genes[i]=="+"):
                                            score += 1
                                        else:
                                            score -= 1 

                if(any(np.isin(c_, i))):
                    if(np.isin(c, self.genes[i])):
                            score += 1
                    else:
                            score -= 1

                if(any(np.isin(d_, i))):
                    if(np.isin(d, self.genes[i])):
                            score += 1
                    else:
                            score -= 1

                if(any(np.isin(e_, i))):    
                    if(len(validar_aux)!=0):
                        if(validar_aux[0]==3 and validar_aux[1]=="+"):
                            if(any(np.isin(e, self.genes[i]))):
                                score += 1
                                elemento_primer_corchete = np.append(elemento_primer_corchete, self.genes[i]) 
                            else: 
                                score -= 1      
                        elif(validar_aux[0]==3 and validar_aux[1]=="-"):    
                            if(any(np.isin(f, self.genes[i]))):
                                score += 1  
                                elemento_primer_corchete = np.append(elemento_primer_corchete, self.genes[i]) 
                            else: 
                                score -= 1      
        
                if(any(np.isin(f_, i))):
                    
                    if(len(validar_aux)!=0):
                        if(validar_aux[0]==3 and validar_aux[1]=="-"):
                            if(any(np.isin(e, self.genes[i]))):
                                score += 1
                                elemento_segundo_corchete = np.append(elemento_segundo_corchete, self.genes[i])
                            else: 
                                score -= 1     
                        elif(validar_aux[0]==3 and validar_aux[1]=="+"):        
                            if(any(np.isin(f, self.genes[i]))):
                                score += 1
                                elemento_segundo_corchete = np.append(elemento_segundo_corchete, self.genes[i])
                            else: 
                                score -= 1   

            if((np.isin("F", elemento_primer_corchete, invert=True))):
                score -= 1
            if((np.isin("F", elemento_segundo_corchete, invert=True))):
                score -= 1
            
            self.fitness = score
            return self.fitness

    def __str__(self):
       return f"Genes: {self.genes}"

    def genes(self):
       return self.genes


class Seleccion:
    
    def __init__(self):
        pass

    def seleccionRanking(self, poblacion):
        
        # Calcular probabilidad de selección
        n = len(poblacion.individuos)
        #El que tiene mayor rango es el de menor distancia 
        fitness = [len(poblacion.individuos[0].genes)-ind.fitness for ind in poblacion.individuos]
        # Obtener los rangos del arreglo
        rangos = rankdata(fitness)
        total = np.sum(rangos)
        probs = rangos / total
        # Seleccionar n padres usando ruleta
        padres = np.random.choice(poblacion.individuos, size=n, replace=True, p=probs)
        return padres

    def seleccionElitismo(self,poblacion):
        padres = np.empty(len(poblacion.individuos), dtype=object)
        for i in range(len(poblacion.individuos)):
            mejores_individuos = sorted(poblacion.individuos, key=lambda x: x.fitness, reverse=True)
            if i < 2:
                padres[i] = mejores_individuos[i]
            else:
                padre1 = random.choice(poblacion.individuos)
                padre2 = random.choice(poblacion.individuos)
                if padre1.fitness < padre2.fitness:
                    padres[i] = padre1
                else:
                    padres[i] = padre2
        return padres            

    def seleccionEstocastica(self, poblacion):
        # Calcular fitness total
        total_fitness = np.sum([len(poblacion.individuos[0].genes)-ind.fitness for ind in poblacion.individuos])
        # Calcular probabilidad de selección
        probs = np.array([((len(poblacion.individuos[0].genes)-ind.fitness) / total_fitness) for ind in poblacion.individuos])
        # Escoger n padres usando selección estocástica
        n_padres = len(poblacion.individuos)
        padres = []
        for i in range(n_padres):
            r = random.random()
            acum_prob = 0
            for j in range(len(probs)):
                acum_prob += probs[j]
                if r < acum_prob:
                    padres.append(poblacion.individuos[j])
                    break
        return padres

    def seleccionRuleta(self, poblacion):
        # Calcular fitness total
        total_fitness = np.sum([len(poblacion.individuos[0].genes)-ind.fitness for ind in poblacion.individuos])
        # Calcular probabilidad de selección
        probs = np.array([((len(poblacion.individuos[0].genes)-ind.fitness) / total_fitness) for ind in poblacion.individuos])
        # Seleccionar n padres usando ruleta
        padres = np.random.choice(poblacion.individuos, size=len(poblacion.individuos), replace=True, p=probs)
        return padres
        
    def seleccionTorneo(self, poblacion):
        padres = np.empty(len(poblacion.individuos), dtype=object)
        for i in range(len(poblacion.individuos)):
            padre1 = random.choice(poblacion.individuos)
            padre2 = random.choice(poblacion.individuos)
            if padre1.fitness < padre2.fitness:
                padres[i] = padre1
            else:
                padres[i] = padre2
        return padres

class Cruce:
    
    def __init__(self, prob_cruce):
        self.prob_cruce = prob_cruce
    
    def cruceBasadoDosPuntos(self, padres):
        hijos = []
        for i in range(0, len(padres)-1, 2):
            padre1 = padres[i]
            padre2 = padres[i+1]
            if random.random() < self.prob_cruce:
                punto1 = random.randint(0, len(padre1.genes) - 2)
                punto2 = random.randint(punto1, len(padre1.genes) - 1)
                
                # Verificar que los puntos de cruce no corten corchetes
                while padre1.genes[punto1:punto2+1].count('[') != padre1.genes[punto1:punto2+1].count(']'):
                    punto1 = random.randint(0, len(padre1.genes) - 2)
                    punto2 = random.randint(punto1, len(padre1.genes) - 1)
                
                hijo1 = Individuo()
                hijo2 = Individuo()
                hijo1.genes = np.concatenate((padre1.genes[:punto1], padre2.genes[punto1:punto2], padre1.genes[punto2:]))
                hijo2.genes = np.concatenate((padre2.genes[:punto1], padre1.genes[punto1:punto2], padre2.genes[punto2:]))
            else:
                hijo1 = padre1
                hijo2 = padre2
            hijos.append(hijo1)
            hijos.append(hijo2)
        return hijos

    
    def cruceBasadoUnPunto(self, padres):
        punto_corte = np.random.randint(1, len(padres[0].genes)-1)
        hijos = []
        for i in range(0, len(padres)-1, 2):
            padre1 = padres[i]
            padre2 = padres[i+1]
            if random.random() < self.prob_cruce:
                hijo1 = Individuo()
                hijo2 = Individuo()
                hijo1.genes = np.append(padre1.genes[:punto_corte], padre2.genes[punto_corte:])
                hijo2.genes = np.append(padre2.genes[:punto_corte], padre1.genes[punto_corte:])
                if hijo1.verificarIndividuo(''.join(hijo1.genes)):
                    hijos.append(hijo1)
                else:
                    hijos.append(padre1)
                if hijo2.verificarIndividuo(''.join(hijo2.genes)):
                    hijos.append(hijo2)
                else:
                    hijos.append(padre2)
            else:
                hijos.append(padre1)
                hijos.append(padre2)
        return np.array(hijos)
    

    def cruceUniforme(self, padres):
            hijos = []
            for i in range(0, len(padres)-1, 2):
                padre1 = padres[i]
                padre2 = padres[i+1]
                if random.random() < self.prob_cruce:
                    hijo1 = Individuo()
                    hijo2 = Individuo()
                    for j in range(len(padre1.genes)):
                        if random.random() < 0.5:
                            hijo1.genes = np.append(hijo1.genes, padre1.genes[j])
                            hijo2.genes = np.append(hijo2.genes, padre2.genes[j])
                        else:
                           hijo1.genes = np.append(hijo1.genes, padre2.genes[j])
                           hijo2.genes = np.append(hijo2.genes, padre1.genes[j])
                    #verificar si las reglas de los hijos son válidas
                    if hijo1.verificarIndividuo(''.join(hijo1.genes)):
                        hijos.append(hijo1)
                    else:
                        hijos.append(padre1)
                    if hijo2.verificarIndividuo(''.join(hijo2.genes)):
                        hijos.append(hijo2)
                    else:
                        hijos.append(padre2)
                else:
                    hijos.append(padre1)
                    hijos.append(padre2)
            return np.array(hijos)

class Mutacion:
    
    def __init__(self, prob_mutacion):
        self.prob_mutacion = prob_mutacion
    
    def mutacionInversion(self, hijos):
        for hijo in hijos:
            if np.random.random() < self.prob_mutacion:
                punto1 = np.random.randint(0, len(hijo.genes) - 2)
                punto2 = np.random.randint(punto1, len(hijo.genes) - 1)
                genes_invertidos = np.flip(hijo.genes[punto1:punto2+1])
                nueva_cadena = ''.join(hijo.genes[:punto1]) + ''.join(genes_invertidos) + ''.join(hijo.genes[punto2+1:])
                if hijo.verificarIndividuo(nueva_cadena):
                    hijo.genes = np.array(list(nueva_cadena))
        return hijos


    def mutacionIntercambio(self, hijos):
        for hijo in hijos:
            if np.random.random() < self.prob_mutacion:
                punto1 = np.random.randint(0, len(hijo.genes) - 1)
                punto2 = np.random.randint(0, len(hijo.genes) - 1)
                aux = hijo.genes.copy()
                aux[punto1], aux[punto2] = aux[punto2], aux[punto1]
                nueva_cadena = ''.join(aux)
                if hijo.verificarIndividuo(nueva_cadena):
                    hijo.genes = np.array(list(nueva_cadena))
        return hijos

        

    def mutacionScramble(self, hijos):
        for hijo in hijos:
            if np.random.random() < self.prob_mutacion:
                punto1 = np.random.randint(0, len(hijo.genes) - 2)
                punto2 = np.random.randint(punto1, len(hijo.genes) - 1)
                genes = hijo.genes[punto1:punto2+1]
                np.random.shuffle(genes)
                nueva_cadena = ''.join(hijo.genes[:punto1]) + ''.join(genes) + ''.join(hijo.genes[punto2+1:])
                if hijo.verificarIndividuo(nueva_cadena):
                    hijo.genes = np.array(list(nueva_cadena))
        return hijos
  
    
    def mutacionFitBit(self, hijos):
        for hijo in hijos:
            if np.random.random() < self.prob_mutacion:
                punto = np.random.randint(0, len(hijo.genes) - 1)
                reemplazo = np.random.choice(['F', '+', '-', '[', ']'])
                genes_mutados = hijo.genes.copy()  # hacemos una copia de los genes del hijo para no modificar el original
                genes_mutados[punto] = reemplazo  # realizamos la mutación en la posición indicada
                if hijo.verificarIndividuo(''.join(genes_mutados)):
                    hijo.genes = genes_mutados  # si el nuevo individuo es válido, lo reemplazamos
        return hijos



class Visualizacion:
    
    def diagrama(self, mejor_individuo):
       genes = mejor_individuo.genes
       # Crear sublistas de dos elementos a partir de genes
       genes_por_coord = [genes[i:i+2] for i in range(0, len(genes), 2)]
       x = np.array([coord[0] for coord in genes_por_coord])
       y = np.array([coord[1] for coord in genes_por_coord])
       plt.plot(x, y, 'ro-')
       plt.plot(x[0], y[0], 'bo')
       plt.title("Mejor solución")
       plt.xlabel("Coordenada X")
       plt.ylabel("Coordenada Y")
       plt.show()

    def diagramaCajas(self, poblacion):
      genes = [i.genes for i in poblacion.individuos]
      fitness = np.array([sum(i) for i in genes])
      # crear el diagrama de cajas
      fig, ax = plt.subplots()
      ax.boxplot(fitness)
      # configurar el título y los ejes
      ax.set_title('Diagrama de cajas')
      ax.set_ylabel('Fitness')
      ax.set_xlabel("Generacion")  
      # mostrar el diagrama
      plt.show() 

    
    def histograma(self, poblacion):
        fits = np.array([ind.fitness for ind in poblacion.individuos])
        plt.hist(fits, bins=20, alpha=0.5)
        plt.title("Distribución de Fitness")
        plt.xlabel("Fitness")
        plt.ylabel("Frecuencia")
        plt.show()
    
    def fitnessGeneraciones(self, mejores_fitness):
        plt.plot(np.array(mejores_fitness))
        plt.title("Mejor Fitness por Generación")
        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.show()

    def fitness_promedio_generaciones(self,promedio,generaciones):
        generaciones = np.array([i for i in range(1,generaciones+1)])
        plt.title("Generaciones VS Fitness")
        plt.xlabel("Numero de Generaciones")
        plt.ylabel("Fitness")
        plt.plot(generaciones,np.array(promedio))
        plt.show()

    def mejor_peor_generaciones(self, peores,mejores,generaciones):
        generaciones_ = np.array([i for i in range(0,generaciones)])
        indices = np.arange(len(generaciones_))
        # Ancho de cada barra
        ancho = 0.35
        plt.bar(indices - ancho/2, np.array(mejores), ancho, label='Mejor fitness')
        plt.bar(indices + ancho/2, np.array(peores), ancho,  label='Peor fitness')
        
        # Ajusta los límites del eje y
        plt.ylim(0, max(max(np.array(peores)), max(np.array(mejores))) + 1)

        # Agrega títulos y etiquetas de eje
        plt.title('Mejor y Peor Fitness por Generación')
        plt.xlabel('Generaciones')
        plt.ylabel('Fitness')
        plt.xticks(indices, generaciones_)
        # Agrega una leyenda
        plt.legend(loc='upper right')
        # Muestra el gráfico
        plt.show()

    def estadisticas(self, poblacion):
        fits = np.array([ind.fitness for ind in poblacion.individuos])
        print(f"Mejor fitness: {np.min(fits)}")
        print(f"Peor fitness: {np.max(fits)}")
        print("Fitness promedio: {} \n\n".format(np.mean(fits)))
    
    def mejorFitness(self, poblacion):
        fitness = np.array(poblacion.calcularFitness())
        return np.min(fitness), fitness

    def peorFitness(self, poblacion):
        return max(poblacion.individuos, key=lambda ind: ind.fitness)    

    def fitnessPromedio(self, poblacion):
        fits = np.array([ind.fitness for ind in poblacion.individuos])
        return np.mean(fits)

    def cadena(self,axioma,produccionR,iteraciones):
      for iteracion in range(0,iteraciones):
        resultado = ""
        for i in (axioma):  
          try:
              resultado += produccionR[i]
          except:
            resultado += i

        axioma=resultado
      return axioma   


    def tortuga(self,genes,size,angulo,angInicial,posicionI,color, name):
        axioma='F'
        produccionR={'F':''.join(genes)}
        print("produccion",produccionR)
        iteraciones=4
        res=self.cadena(axioma,produccionR,iteraciones)
        
        turtle.penup()
        turtle.goto(posicionI)
        turtle.pendown()
        turtle.color(color)
        turtle.right(angInicial) 
        turtle.bgcolor("black")
        
        
        lista = []
        turtle.tracer(False)  # Desactivar la animación
        
        for i in (res):
            try:
                if(i=="F"):
                    turtle.forward(size)
                elif(i=="-"):
                    turtle.left(angulo) 
                elif(i=="+"):
                    turtle.right(angulo)
                elif(i=="["):
                    lista.append((turtle.heading(), turtle.position()))
                elif(i=="]"):
                    heading, position = lista.pop()
                    turtle.penup()
                    turtle.goto(position)
                    turtle.setheading(heading)
                    turtle.pendown()
                else:
                    pass
            except:
                break
        
        turtle.screensize()
        # Pone el fondo en negro
        
        turtle.update()  # Mostrar el resultado final
        canvas = turtle.getcanvas()

        # Guarda el canvas como imagen en formato PostScript
        postscript = canvas.postscript(colormode='color')

        # Convierte la imagen PostScript a una imagen en formato PNG
        image = Image.open(io.BytesIO(postscript.encode('utf-8')))
        nombre=str(name)+".png"
        image.save(nombre, 'png')


        
        turtle.clear()
        
    def crearGif(self,generaciones,image_files):

            import imageio.v2 as imageio
            import os

            # Lista para almacenar las imágenes
            images = []

            # Iterar sobre los archivos de imagen y agregarlos a la lista
            for filename in image_files:
                images.append(imageio.imread(filename))

            # Guardar las imágenes como un archivo GIF
            imageio.mimsave('mi_animacion.gif', images, duration=5) 

        


class AlgoritmoGenetico:
    
    def __init__(self, tam_poblacion, num_generaciones, prob_cruce, prob_mutacion,opcionSeleccion,opcionCruce,opcionMutacion,n_genes,pasos, angulo, angI, posI, color):
        self.tam_poblacion = tam_poblacion
        self.num_generaciones = num_generaciones
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.poblacion = None
        self.seleccion = Seleccion()
        self.cruce = Cruce(prob_cruce)
        self.mutacion = Mutacion(prob_mutacion)
        self.visualizacion = Visualizacion()
        self.opcionSeleccion=opcionSeleccion
        self.opcionCruce = opcionCruce
        self.opcionMutacion = opcionMutacion
        self.pasos=pasos
        self.angulo=angulo
        self.angInicial=angI
        self.posicionI=posI
        self.color=color
        self.n_genes=n_genes
    
    def ejecutar(self):
        opcionesSeleccion = [1,2,3,4,5]
        opcionesCruce = [1,2,3]
        opcionesMutacion = [1,2,3,4]
        okay = True
        generacionesNecesarias = 0
        images = []
       
        # Inicializar población
        self.poblacion = Poblacion(self.tam_poblacion,self.n_genes)
        self.poblacion.inicializarPoblacion()

        # Calcular fitness de la población inicial
        #self.poblacion.calcularFitness()
        self.poblacion.calcularFitness()
        
        mejores_fitness = []
        peores_fitness = []
        fitness_promedio_generacion = []
        image_files = []

        if((self.opcionSeleccion in opcionesSeleccion)==False):
               print("Opcion {} para seleccion no es valida".format(self.opcionSeleccion))
               okay = False
        if((self.opcionCruce in opcionesCruce)==False):
               print("Opcion {} para cruce no es valida".format(self.opcionCruce))
               okay = False
        if((self.opcionMutacion in opcionesMutacion)==False):
               print("Opcion {} para seleccion no es valida".format(self.opcionMutacion))
               okay = False    

        if(okay==True):
            # Iterar sobre todas las generaciones
            for i in range(self.num_generaciones):
                
                # print(self.poblacion)
                print("\n\n==================================================")
                print("\t\t    Generacion",i)
                print("==================================================\n")
                # genes = [i.genes for i in self.poblacion.individuos]
                # print("Genes",genes)
                
                #fitness = [i.fitness for i in self.poblacion.individuos]
                #print("Fitness",fitness)

                #for i in range(0,len(self.poblacion.individuos)):
                  #print("Objetivo",self.objetivo)
                  #print("Genes    {} Fitness {}".format(genes[i],fitness[i]))
                
                ##print("Mejor fitness",min(self.poblacion.fitness))


                ##fitness = [sum(i) for i in genes]
                #print("Fitness",fitness)
                # print("Promedio",sum(fitness)/len(self.poblacion.individuos))
                ##fitness_promedio_generacion.append(sum(fitness)/len(self.poblacion.individuos))

                
                # Seleccionar padres
                if(self.opcionSeleccion == 1):
                  padres = self.seleccion.seleccionRuleta(self.poblacion)
                elif(self.opcionSeleccion == 2):
                  padres = self.seleccion.seleccionEstocastica(self.poblacion)
                elif(self.opcionSeleccion == 3):
                  padres = self.seleccion.seleccionRanking(self.poblacion)
                elif(self.opcionSeleccion == 4):
                   padres = self.seleccion.seleccionTorneo(self.poblacion)
                elif(self.opcionSeleccion == 5):
                   padres = self.seleccion.seleccionElitismo(self.poblacion)

                # cruce padres y crear hijos
                if(self.opcionCruce == 1):
                  hijos = self.cruce.cruceBasadoUnPunto(padres)
                elif(self.opcionCruce == 2):
                  hijos = self.cruce.cruceBasadoDosPuntos(padres)
                elif(self.opcionCruce == 3):
                  hijos = self.cruce.cruceUniforme(padres)


                if(self.opcionMutacion == 1):
                  hijos = self.mutacion.mutacionFitBit(hijos)
                elif(self.opcionMutacion == 2):
                  hijos = self.mutacion.mutacionIntercambio(hijos)
                elif(self.opcionMutacion == 3):
                  hijos = self.mutacion.mutacionScramble(hijos)
                elif(self.opcionMutacion == 4):
                  hijos = self.mutacion.mutacionInversion(hijos)
                # Reemplazar población anterior con hijos

    
                self.poblacion.individuos = hijos
                
                # Calcular fitness de la nueva población
                ##self.poblacion.calcularFitness()
                self.poblacion.calcularFitness()
                # Obtener el mejor fitness de la generación actual
                #mejor_fitness_generacion = max(self.poblacion.fitness)

                mejor_fitness_generacion = min(self.poblacion.fitness)
                
                print("Fitness_generacion", mejor_fitness_generacion)
                # Guardar el mejor fitness de la generación actual
                mejores_fitness.append(mejor_fitness_generacion)
                best_in = self.poblacion.getMejorIndividuo()

                #peor_fitness_generacion = min(self.poblacion.fitness)
                peor_fitness_generacion = max(self.poblacion.fitness)
                peores_fitness.append(peor_fitness_generacion)
                peor = self.visualizacion.peorFitness(self.poblacion)
                


                #nombre=str(i)
                #self.visualizacion.tortuga(best_in.genes,self.pasos,self.angulo,self.angInicial,self.posicionI,self.color,nombre)
                #image_files.append(nombre+".png")

                if(min(self.poblacion.fitness)==0):
                  break
            
                print(''.join(best_in.genes))
            # self.visualizacion.mostrarMejorIndividuo(best_in,self.alto,self.ancho)
            # print("\nPeores Fitness: ",peores_fitness)
            print("Mejores Fitness: ",mejores_fitness)
            print("Mejor Individuo",''.join(best_in.genes))
            print("\n")
            #self.visualizacion.mejor_peor_generaciones(peores_fitness,mejores_fitness,generacionesNecesarias)
            # print("\n\n")
            # #print("Fitness promedio",fitness_promedio_generacion)
            # print("\n\n")
            #self.visualizacion.fitness_promedio_generaciones(fitness_promedio_generacion,self.num_generaciones)
              
                
            #self.visualizacion.tortuga(best_in.genes,self.pasos,self.angulo,self.angInicial,self.posicionI,self.color)
            #ruta=os.getcwd()
            #self.visualizacion.crearGif(self.num_generaciones,image_files)

"""## Primer experimento"""

tam_poblacion = 1000
num_generaciones = 100
prob_cruce = 0.9
prob_mutacion = 0.01

opcionSeleccion = 4
opcionCruce = 1
opcionMutacion = 1 

pasos=20
angulo=20
angI=90
n_genes=19

posI=(30, 10)
color='coral'


#["seleccionRuleta","seleccionEstocastica","seleccionRanking","seleccionTorneo"]
#["cruceBasadoUnPunto","cruceBasadoDosPuntos","cruceUniforme"]
#["mutacionFitBit","mutacionIntercambio","mutacionScramble","mutacionInversion"]

algoritmoG=AlgoritmoGenetico(tam_poblacion, num_generaciones, prob_cruce, prob_mutacion,opcionSeleccion,opcionCruce,opcionMutacion,n_genes,pasos, angulo, angI, posI, color)
algoritmoG.ejecutar()