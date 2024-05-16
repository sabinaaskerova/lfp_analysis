import numpy as np
import matplotlib.pyplot as plt
import os 

class Calculate_PSD:
    @staticmethod
    def compute_PSD_all_portions(data_path, ana,stype,file,fmin,fmax,sliding_window_length,overlap,n_channels=14,portion_length=10,result_folder='psd_pca/',
                                    log=False,mean=False):        
            label_list = []
            from scipy import fft
            
            df = 1/sliding_window_length # frequency resolution
            fmax_num = int((fmax)/df) # number of frequency points
            fmin_num = int(fmin/df)
            
            label_list = []
            print(stype)
            label = data_path+file+"_"+ana+"_"+stype+"_new.dat"
            print("datafile:",label)
            label_list.append(stype)
            data=np.loadtxt(label)
            num_time = np.shape(data)[0]
            num_chans = np.shape(data)[1]
            
            
            fs = 500.0 #%%%%%%%%%%%%%%%
            dt = 1.0/fs #%%%%%%%%%%%%%%%%
            #dt = 600/num_time
            #onesec_length = int(1/dt)
            overlap_num = int(overlap/dt)
            nb_points_window = int(sliding_window_length/dt) # number of points in the window 5/0.002 = 2500
            power_ = np.zeros(nb_points_window//2) 

            freqs = np.linspace(fmin,fmax,fmax_num-fmin_num)
            #max_time_index = int(num_time/onesec_length - sliding_window_length)
            #for j in range(0, max_time_index): 
            
            #outlabel = result_folder+'ascii_out_'+file+'_'+ana+'_'+stype+'_'+str(fmin)+'_'+str(fmax)+'/'+'out%d.dat'%(j)
            #if not os.path.exists(result_folder+'ascii_out_'+file+'_'+ana+'_'+stype+'_'+str(fmin)+'_'+str(fmax)+'/'):
            #        os.makedirs(result_folder+'ascii_out_'+file+'_'+ana+'_'+stype+'_'+str(fmin)+'_'+str(fmax)+'/')

            #start_time = j*onesec_length
            #end_time = (j+1)*onesec_length+portion_length*onesec_length
            #start_time = j*onesec_length
            #end_time = (j+1)*onesec_length+portion_length*onesec_length
            #print("start_time, end_time", start_time, end_time)

            start_time = 0
            end_time = num_time
            duration = (end_time-start_time)*dt
            num_windows = int((duration-sliding_window_length)/overlap)
            time = np.linspace(start_time*dt,end_time*dt,num_windows)
            PSD = np.zeros((n_channels, len(freqs),num_windows))
        
            for i in range(num_chans):
                y = data[start_time:end_time,i]
                if mean == True:
                    y -= np.mean(y)
                    
                power_[:]=0.0
                for num in range(num_windows):
                    y_window = y[num*overlap_num:num*overlap_num+nb_points_window]
                    signal_fft0 = fft.fft(y_window) # Welch PSD
                    power_ =np.abs(signal_fft0[:nb_points_window//2])**2/float(nb_points_window)
                    if log == False:
                        power           = power_[fmin_num:fmax_num]/float(num_windows)
                    else:
                        power           = np.log(power_[fmin_num:fmax_num]/float(num_windows))
                    PSD[i,:,num] = power
                
                #ff = open(outlabel,'w+')
                #res_str = ''
                #for j in range(0, fmax_num-fmin_num):
                #    res_str = '%f '%freqs[j]
                #    for k in range(0, num_chans):
                #        res_str += '%f  '%PSD[k,j] 
                #    res_str += '\n'
                #    ff.write(res_str)
                #ff.close()
                #print("written ",outlabel)
            
            return time,freqs,PSD
    
    @staticmethod
    def bands(freqs,PSD,fmin,fmax):
        df = freqs[1]-freqs[0]
        fmin_num = int(fmin/df)
        fmax_num = int(fmax/df)
        power = PSD[:,fmin_num:fmax_num,:]
        freq = freqs[fmin_num:fmax_num]
        mean_power = np.mean(power,axis=1)
        return freq,power,mean_power
    
    @staticmethod
    def plot_colored_psd(animal, ana,stype,freqlabel,time,freqs,PSD,psd_file_prefix, channel_no=1, plot=False):
    #def plot_colored_psd(data_path, animal, ana,stype, fmin, fmax, sliding_window_length, psd_file_prefix, df, channel_no=1, plot=False):
        #label = data_path+animal+"_"+ana+"_"+stype+"_new.dat"
        #data=np.loadtxt(label)
        
        ## PSD: n_channels, len(freqs),num_windows
        num_time = np.shape(PSD)[2]
        tmin = time[0]
        tmax = time[-1]
        fmax = freqs[-1]
        fmin = freqs[0]

        dt = time[1]-time[0] #%%%%%%%%%%%%%%%%
        #dt = 600/num_time
        #onesec_length = int(1/dt)
        df = freqs[1]-freqs[0]#1/sliding_window_length
        #max_time_index = int(num_time/onesec_length - sliding_window_length)
        
        def forceAspect(ax,aspect=1):
            im = ax.get_images()
            extent =  im[0].get_extent()
            ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
        
        fmax_num = len(freqs)#int((fmax-fmin)/df) 
        #pow_arr = np.zeros((max_time_index, fmax_num))

        
        #for i in range(0,max_time_index):
            #print("####### ",psd_file_prefix+'ascii_out_'+animal+'_'+ana+'_'+stype+'_'+str(fmin)+'_'+str(fmax)+'/out%d.dat'%i)
        #    pow_arr[i] = np.loadtxt(psd_file_prefix+'ascii_out_'+animal+'_'+ana+'_'+stype+'_'+str(fmin)+'_'+str(fmax)+'/out%d.dat'%i)[:,channel_no]
        plt.figure('%d_channel'%channel_no,figsize=(12, 4), facecolor='white')
        c = plt.imshow(PSD[channel_no,:,:], interpolation ='none', origin ='lower', extent=[tmin,tmax,fmin, fmax]) 
        
        plt.colorbar(c, label='Power Spectral Density')
        plt.xlabel('Time')
        plt.ylabel('Frequency indexes(%f-%f Hz)'%(fmin, fmax))
        plt.title('%s-PSD over Time on PCA '%freqlabel+animal+" "+ana+" "+stype+" "+str(channel_no))
        forceAspect(plt.gca(),aspect=4)

        string = freqlabel+'_'+animal+'_'+ana+'_'+stype+'_'+str(fmin)+'_'+str(fmax)
        graphs_path = psd_file_prefix+'ascii_out_'+string+"/graphs/"
        if not os.path.exists(graphs_path):
            os.makedirs(graphs_path)
        plt.savefig(graphs_path+string+".png")
        print("Saved the graph to "+graphs_path+string+".png")
        if plot:
            plt.show()
        plt.close()
          


        