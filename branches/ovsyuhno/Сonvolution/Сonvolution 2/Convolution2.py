
import sys
import argparse
 
def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-i', '--inputFileName', type = str)
    parser.add_argument ('-o', '--outputFileName', type = str)
    #parser.add_argument ('-s', '--surface', type = str)
    parser.add_argument ('-l', '--convolutionSetting', type = float)
    parser.add_argument ('-t', '--step', type = float)
    parser.add_argument ('-r', '--radius', type = float)
    parser.add_argument ('-n', '--numberOfComputations', type = int)
    
    return parser
        
if __name__ == '__main__':
    parser = createParser()
    args = parser.parse_args()
    i = 0
    #print('D:' + '//'[:-1] + chr(0x78) + 'mcmlLauncher.exe', '-i', args.inputFileName, '-o', str(i) + '.mcml.out', '-s', args.surface, '-x', str(args.step * i), '-y', str(0))
    
   # while i < args.numberOfComputations:
        #print i
        #import subprocess
        #subprocess.call(['xmcmlLauncher.exe', '-i', args.inputFileName, '-o', str(i) + '.mcml.out', '-s', args.surface, '-x', str(args.step * i), '-y', str(0)])
        #i = i + 1
    import subprocess
    subprocess.call(['Convolution.exe', '-i', inputFileName + str(0) + '.mcml.out', '-o', args.outputFileName, '-l', str(args.convolutionSetting), '-w', '0', '-d', '0'])
    i = 1
    #print('D:\Convolution.exe', '-i', str(0) + '.mcml.out', '-o', args.outputFileName, '-l', str(args.convolutionSetting), '-w', '1', '-d', '0')
    while i < args.numberOfComputations:
        print i
        import subprocess
        subprocess.call(['Convolution.exe', '-i', inputFileName + str(i) + '.mcml.out', '-o', args.outputFileName, '-l', str(args.convolutionSetting), '-w', '1', '-d', '0'])
        i = i + 1

