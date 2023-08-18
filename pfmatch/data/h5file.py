import h5py as h5
import numpy as np
from pfmatch.flashmatch_types import QCluster, Flash

class H5File(object):
    '''
    Class to interface with the HDF5 dataset for pfmatch. 
    It can store arbitrary number of events.
    Each event contains arbitrary number of QCluster and Flash using a "ragged array".
    '''
    
    def __init__(self,file_name,mode):
        '''
        Constructor
            file_name: string path to the data file to be read/written
            mode: h5py.File constructor "mode" (w=write, a=append, r=read)
        '''
        dt_float = h5.vlen_dtype(np.dtype('float32'))
        dt_int   = h5.vlen_dtype(np.dtype('int32'))
        self._mode = mode
        self._f = h5.File(file_name,self._mode)
        if self._mode in ['w','a']:
            self._wh_point = self._f.create_dataset('point', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_group = self._f.create_dataset('group', shape=(0,), maxshape=(None,), dtype=dt_int  )
            self._wh_flash = self._f.create_dataset('flash', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_point.attrs['note'] = '"point": 3D points with photon count. Reshape to (N,4) for total of N points. See "group" attributes to split into cluster.'
            self._wh_group.attrs['note'] = '"group": An array of integers = number of points per cluster. The sum of the array == N points of "point" data.'
            self._wh_flash.attrs['note'] = '"flash": Flashes (photo-electrons-per-pmt). Reshape to (K,180) for K flashes' 
            
            # Attributes below are not implemented
            self._wh_qtime = self._f.create_dataset('qtime', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_qtime_true = self._f.create_dataset('qtime_true', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_ftime = self._f.create_dataset('ftime', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_ftime_true = self._f.create_dataset('ftime_true', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_ftime_width = self._f.create_dataset('ftime_width', shape=(0,), maxshape=(None,), dtype=dt_float)

    def close(self):
        self._f.close()
        
    def __del__(self):
        try:
            self._f.close()
        except:
            pass
        
    def __len__(self):
        return len(self._f['point'])
    
    def __getitem__(self,idx):
        return self.read_one(idx)
    
    def __str__(self):
        msg=f'{len(self)} entries in this file. Raw hdf5 attribute descriptions below.\n'
        for k in self._f.keys():
            try:
                msg += self._f[k].attrs['note']
                msg += '\n'
            except:
                pass
        return msg
        
    def read_one(self,idx):
        '''
        Read one event specified by the integer index
        '''
        qcluster_vv,flash_vv = self.read_many([idx])
        if len(qcluster_vv)<1:
            return
        return (qcluster_vv[0],flash_vv[0])
            
    def read_many(self,idx_v):
        '''
        Read many event specified by an array of integer indexes
        '''
        flash_vv = []
        qcluster_vv = []
        
        for idx in idx_v:
            if idx >= len(self):
                print('Cannot access entry',idx,'(out of range)')
                return
        
        event_point_v = [np.array(data).reshape(-1,4) for data in self._f['point'][idx_v]]
        event_group_v = self._f['group'][idx_v]
        event_flash_v = [np.array(data) for data in self._f['flash'][idx_v]]
        
        for i in range(len(idx_v)):
            
            event_point = event_point_v[i]
            event_group = event_group_v[i]
            event_flash = event_flash_v[i]
            
            flash_v = []
            qcluster_v = []
            
            event_flash = event_flash.reshape(-1,180)
            for f in event_flash:
                flash_v.append(Flash())
                flash_v[-1].fill(f)

            start = 0
            for ctr in event_group:
                qcluster_v.append(QCluster())
                qcluster_v[-1].fill(event_point[start:start+ctr])
                start = start + ctr
            
            qcluster_vv.append(qcluster_v)
            flash_vv.append(flash_v)
            
        return (qcluster_vv, flash_vv)
        
    
    def write_one(self,qcluster_v,flash_v):
        '''
        Write many event to a file with the provided list of QCluster and Flash
        '''
        self.write_many([qcluster_v],[flash_v])
        
    def write_many(self,qcluster_vv,flash_vv):
        '''
        Write many event to a file with the provided list of QCluster and Flash
        '''
        if not self._mode in ['w','a']:
            print('ERROR: the dataset is not created in the w (write) nor a (append) mode')
            return
        if not len(qcluster_vv) == len(flash_vv):
            print('ERROR: the length of qcluster_vv',len(qcluster_vv),'is not same as flash_vv',len(flash_vv))
            return        
        
        # expand the output count by one for the new entry
        data_index = self._wh_point.shape[0]
        data_count = data_index+len(qcluster_vv)
        self._wh_point.resize(data_count,axis=0)
        self._wh_group.resize(data_count,axis=0)
        self._wh_flash.resize(data_count,axis=0)

        for i in range(len(qcluster_vv)):
        
            qcluster_v = qcluster_vv[i]
            flash_v = flash_vv[i]
            
            ntracks = len(qcluster_v)
            nflash  = len(flash_v)

            point_v  = []
            for j in range(ntracks):
                point_v.append(qcluster_v[j].qpt_v.cpu().numpy())
            point_group = np.array([pts.shape[0] for pts in point_v])
            point_v = np.concatenate(point_v)

            photon_v = []
            photon_err_v = []
            for j in range(nflash):
                photon_v.append(flash_v[j].pe_v.cpu().numpy())
                photon_err_v.append(flash_v[j].pe_err_v.cpu().numpy())
            photon_v = np.concatenate(photon_v)
            photon_err_v = np.concatenate(photon_err_v)
            
            self._wh_point[data_index] = point_v.flatten()
            self._wh_group[data_index] = point_group
            self._wh_flash[data_index] = photon_v.flatten()
            
            data_index+=1

def test_H5File():
    '''
    Function to test H5File class (True=success, False=broken)
    '''
        
    input_data=[]

    # Create a file with fake data
    f=H5File('test.h5','w')
    nevents = int(np.random.random()*100)
    for i in range(nevents):
        ntracks = int(np.random.random()*10)+1
        qcluster_v = []
        for j in range(ntracks):
            qcluster_v.append(QCluster())
            qcluster_v[-1].fill(np.random.random(size=(int(np.random.random()*100)+1,4)))
        nflash = int(np.random.random()*10)+1 
        flash_v = []
        for j in range(nflash):
            flash_v.append(Flash())
            flash_v[-1].fill(np.random.random(180))
        
        f.write_one(qcluster_v,flash_v)
        qcluster_v = [qc.qpt_v.cpu().numpy() for qc in qcluster_v]
        flash_v = [f.pe_v.cpu().numpy() for f in flash_v]
        input_data.append((qcluster_v,flash_v))

    f.close()
    
    # Read the data file and check it's identical to the generated input
    f=H5File('test.h5','r')
    stored_data=[]
    for i in range(len(f)):
        qcluster_v,flash_v = f.read_one(i)
        qcluster_v = [qc.qpt_v.cpu().numpy() for qc in qcluster_v]
        flash_v = [f.pe_v.cpu().numpy() for f in flash_v]
        stored_data.append((qcluster_v,flash_v))
        
    # Compare
    flag=True
    if not len(input_data) == len(stored_data):
        print('Error: written v.s. read event count:', len(input_data),'!=',len(stored_data))
        flag=False
    for i in range(len(input_data)):
        qin,fin=input_data[i]
        qout,fout=stored_data[i]
        
        if not len(qin) == len(qout):
            print('Error: entry',i,'shape mismatch for QCluster',len(qin),'!=',len(qout))
            flag=False
            
        if not len(fin) == len(fout):
            print('Error: entry',i,'shape mismatch for Flash',len(fin),'!=',len(fout))
            flag=False
            
        qin_sum  = np.sum([qc.sum() for qc in qin ])
        qout_sum = np.sum([qc.sum() for qc in qout])
        if not abs(qin_sum-qout_sum) < max(abs(qin_sum/1.e5),abs(qout_sum/1.e5)):
            print('Error: entry',i,'value sum mismatch for QCluster',qin_sum,'!=',qout_sum)
            flag=False
            
        fin_sum  = np.sum([f.sum() for f in fin ])
        fout_sum = np.sum([f.sum() for f in fout])
        if not abs(fin_sum-fout_sum) < max(abs(fin_sum/1.e5),abs(fout_sum/1.e5)):
            print('Error: entry',i,'value sum mismatch for Flash',fin_sum,'!=',fout_sum)
            flag=False
            
    return flag