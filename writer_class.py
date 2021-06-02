import h5py
import numpy as np
import datetime
import pathlib


class writer():

    def __init__(self):
        pass

    def writeHDF5_2D(self,phi_list,slist,PhysSize,fname,material_string,author='PJD',igormodelname='4DSTEM'):
        ''' Writes list of phi and s to hdf5 file. Uses old (<=v0.8) Cy-RSoXS
            convention

            TO DO: add in support for new convention and ability to switch
            between versions automatically '''
        print(f'--> Marking {fname}')
        with h5py.File(fname,'w') as f:
                num_mat = len(phi_list)
                f.create_dataset("igor_parameters/igormaterialnum",data=float(num_mat))
                i = 1
                for phi, s in zip(phi_list,slist):
                    f.create_dataset(f"vector_morphology/Mat_{i}_alignment",data=s,compression='gzip',compression_opts=9)
                    f.create_dataset(f"vector_morphology/Mat_{i}_unaligned",data=phi,compression='gzip',compression_opts=9)
                    i += 1

                f.create_dataset('morphology_variables/creation_date', data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
                f.create_dataset('morphology_variables/film_normal', data=[1,0,0])
                f.create_dataset('morphology_variables/morphology_creator', data=author)
                f.create_dataset('morphology_variables/name', data=author)
                f.create_dataset('morphology_variables/version', data=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
                f.create_dataset('morphology_variables/voxel_size_nm', data=float(PhysSize))

                f.create_dataset('igor_parameters/igorefield', data="0,1")
                f.create_dataset('igor_parameters/igormaterials', data=material_string)
                f.create_dataset('igor_parameters/igormodelname', data="4DSTEM")
                f.create_dataset('igor_parameters/igormovie', data=0)
                f.create_dataset('igor_parameters/igorname', data="perp001")
                f.create_dataset('igor_parameters/igornum', data=0)
                f.create_dataset('igor_parameters/igorparamstring', data="n/a")
                f.create_dataset('igor_parameters/igorpath', data="n/a")
                f.create_dataset('igor_parameters/igorrotation', data=0)
                f.create_dataset('igor_parameters/igorthickness', data=1)
                f.create_dataset('igor_parameters/igorvoxelsize', data=1)
        return fname

    def write_config(self,fname,startEnergy, endEnergy, incrementEnergy,
                    startAngle,endAngle, incrementAngle, numThreads,
                    numX, numY, numZ, PhysSize):

        f = open(fname, "w")
        f.write("StartEnergy = " + str(float(startEnergy)) + ";\n");
        f.write("EndEnergy = " + str(float(endEnergy)) + ";\n");
        f.write("IncrementEnergy = " + str(float(incrementEnergy)) + ";\n");
        f.write("StartAngle = " + str(float(startAngle)) + ";\n");
        f.write("EndAngle = " + str(float(endAngle)) + ";\n");
        f.write("IncrementAngle = " + str(float(incrementAngle)) + ";\n");
        f.write("NumThreads = " + str(int(numThreads)) + ";\n");
        f.write("NumX = " + str(int(numX)) + ";\n");
        f.write("NumY = " + str(int(numY)) + ";\n");
        f.write("NumZ = " + str(int(numZ)) + ";\n");
        f.write("PhysSize = " + str(float(PhysSize)) + ";\n");
        f.close();

    def find_nearest(self,array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    def get_interpolated_value(self,array,value,nearest_id,energy_id):
        valArray = np.zeros(array.shape[1]);
        if(array[nearest_id][energy_id] > value):
            xp = [array[nearest_id][energy_id], array[nearest_id - 1][energy_id]];
            for i in range(1,len(valArray)):
                yp = [array[nearest_id][i], array[nearest_id - 1][i]];
    #             print(xp,yp)
                valArray[i] = np.interp(value,xp,yp);

        elif (array[nearest_id][energy_id] < value):
            xp = [array[nearest_id][energy_id], array[nearest_id + 1][energy_id]];
            for i in range(1,len(valArray)):
                yp = [array[nearest_id][i], array[nearest_id + 1][i]];
    #             print(xp,yp)
                valArray[i] = np.interp(value,xp,yp);

        else:
            for i in range(1,len(valArray)):
                valArray[i] = value[nearest_id][i];

        valArray[energy_id] = value;
    #     print(valArray)
        return valArray;

    def dump_dataVacuum(self,index,energy,f):
        Header = "EnergyData" + str(index) +":\n{\n";
        f.write(Header);
        Energy = "Energy = " + str(energy) + ";\n";
        f.write(Energy);
        BetaPara = "BetaPara = " + str(0.0) + ";\n";
        f.write(BetaPara);
        BetaPerp = "BetaPerp = " + str(0.0) + ";\n";
        f.write(BetaPerp);
        DeltaPara = "DeltaPara = " + str(0.0) + ";\n";
        f.write(DeltaPara);
        DeltaPerp = "DeltaPerp = " + str(0.0) + ";\n";
        f.write(DeltaPerp);
        f.write("}\n");



    def dump_data(self,valArray,index,labelEnergy,f):
        Header = "EnergyData" + str(index) +":\n{\n";
        f.write(Header);
        Energy = "Energy = " + str(valArray[labelEnergy["Energy"]]) + ";\n";
        f.write(Energy);
        BetaPara = "BetaPara = " + str(valArray[labelEnergy["BetaPara"]]) + ";\n";
        f.write(BetaPara);
        BetaPerp = "BetaPerp = " + str(valArray[labelEnergy["BetaPerp"]]) + ";\n";
        f.write(BetaPerp);
        DeltaPara = "DeltaPara = " + str(valArray[labelEnergy["DeltaPara"]]) + ";\n";
        f.write(DeltaPara);
        DeltaPerp = "DeltaPerp = " + str(valArray[labelEnergy["DeltaPerp"]]) + ";\n";
        f.write(DeltaPerp);
        f.write("}\n");




    def write_materials(self, startEnergy, endEnergy,increment,dict,labelEnergy,numMaterial):
        NumEnergy = int(np.round((endEnergy - startEnergy)/increment + 1));
        # force types
        startEnergy = float(startEnergy)
        endEnergy = float(endEnergy)
        increment = float(increment)
        numMaterial = int(numMaterial)

        for numMat in range(0,numMaterial):
            f = open("Material" + str(numMat) + ".txt", "w")
            fname = dict["Material" + str(numMat)]
            if(fname != 'vacuum'):
                Data = np.loadtxt(fname,skiprows=1);
                Data = Data[Data[:,labelEnergy["Energy"]].argsort()]
                for i in range(0,NumEnergy):3
                    currentEnergy = startEnergy + i* increment;
                    nearest_id = self.find_nearest(Data[:,labelEnergy["Energy"]],currentEnergy)
                    ValArray = self.get_interpolated_value(Data,currentEnergy,nearest_id,labelEnergy["Energy"])
                    self.dump_data(ValArray,i,labelEnergy,f)

            else:
                for i in range(0,NumEnergy):
                    energy = startEnergy + increment*i
                    self.dump_dataVacuum(i,energy,f)
            f.close()
