# TIME WIZARD

time = "7:10:26"
epochs = 21
times = time.split(":")

totaltime = ((int)(times[0]))*60*60 + ((int)(times[1]))*60 + ((int)(times[2]))
newtime = (int) (totaltime/epochs)

import datetime
final = str(datetime.timedelta(seconds=newtime))

print(final)