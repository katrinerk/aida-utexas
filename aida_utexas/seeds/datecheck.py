import datetime

#########
# AIDA temporal info handling,
# with missing entries for days/months/years
class AidaIncompleteDate:
    def __init__(self, year, month, day):
        def checkme(date, datestr):
            if date is not None and not isinstance(date, int):
                if isinstance(date, str) and date.isdigit():
                    date = int(date)
                else:
                    if not(isinstance(date, str) and (date.strip("X") == "")):
                        print("unexpected type for", datestr,  type(year))
                    date = None
            return date

        year = checkme(year, "year")
        month = checkme(month, "month")
        day = checkme(day, "day")
        
        self.year = year
        self.month = month
        self.day = day


    def to_python_date(self):
        if self.year is None:
            thisyear = 0
        else:
            thisyear = self.year
        if self.month is None:
            thismonth = 1
        else:
            thismonth = self.month
        if self.day is None:
            thisday = 1
        else:
            thisday = self.day

        return datetime.date(thisyear, thismonth, thisday)
        
    def last_day_of_month(self):
        if self.day is None:
            return False

        testdate = self.to_python_date()
        testdate2 = self.to_python_date() + datetime.timedelta(1)
        if testdate2.month != testdate.month:
            return True

        return False
            
    # add a day to the date
    def add_day(self):
        if self.day is None:
            return AidaIncompleteDate(self.year, self.month, self.day)

        testdate = self.to_python_date()
        testdate2 = self.to_python_date() + datetime.timedelta(1)

        newday = testdate2.day
        if self.month is not None:
            newmonth = testdate2.month
        else:
            newmonth = None
        if self.year is not None:
            newyear = testdate2.year
        else:
            newyear = None
        return AidaIncompleteDate(newyear, newmonth, newday)

    def is_before(self, date2, add_a_day = False):
        if add_a_day:
            date2 = date2.add_day()
        # print("HIERbefore", self.year, self.month, self.day, date2.year, date2.month, date2.day)
        
        if self.year is None or date2.year is None:
            # we don't know which year one of the dates given is:
            # then the 2nd can definitely be before
            return True

        elif self.year > date2.year:
            # t1 is in a later year, so not before t2
            return False

        elif self.year < date2.year:
            # t1 is in an earlier year than t2, so definitely before t2
            return True

        ###
        # same year, move on to month
        if self.month is None and date2.month is None:
            # we don't know the month in either, so t1 can always be before t2
            return True
        
        elif self.month is None and date2.month > 1:
            # we dont know t1's month, and t2 is in Feb at the earliest
            return True
        
        elif date2.month is None and self.month < 12:
            # we don't know t2's month, and t1 is November at the latest
            return True
        
        elif self.month is not None and date2.month is not None and self.month < date2.month: 
            # t1 earlier month than t2, so definitely before
            return True

        elif self.month is not None and date2.month is not None and self.month > date2.month:
            # t1 later month than t2, so definitely not before
            return False

        ###
        # we need to compare days

        if self.day is None and date2.day is None:
            return True
        
        if self.day is not None and date2.day is not None and self.day < date2.day:
            return True
        
        elif date2.day is not None and self.day is None and date2.day > 2:
            # t1, t2 can be same year and month. if t2.day is at least 2, t2 can always be before t1
            return True
        
        elif self.day is not None and date2.day is None:
            if self.last_day_of_month():
                return False
            else:
                return True

        # comparison of days showed that t1 cannot be before t2
        return False


        
    # time comparisons:
    # can t1 be the same as  t2?
    def is_eq(self, date2, add_a_day = False):
        if add_a_day:
            # try adding a day either way
            newdate2 = date2.add_day()
            # print("HIER1", newdate2.year, newdate2.month, newdate2.day, "vs", self.year, self.month, self.day)
            if self.is_eq(newdate2):
                return True
            newself = self.add_day()
            # print("HIER2", newself.year, newself.month, newself.day, "vs", date2.year, date2.month, date2.day)
            if newself.is_eq(date2):
                return True

        # print("HIEReq", self.year, self.month, self.day, date2.year, date2.month, date2.day)            
        
        # try exact match
        if self.year is not None and date2.year is not None and self.year != date2.year:
            return False

        # at this point we know the year can be compatible.
        
        if self.month is not None and date2.month is not None and self.month != date2.month:
            return False

        # at this point we know the month can be compatible.
        if self.day is not None and date2.day is not None and self.day != date2.day:
            return False

        # at this point, the whole dates can be compatible
        return True

################################################
    
# returns true if this statement has no temporal constraint
# or matches the temporal constraint
# leeway levels: 0 = no leeway
# 1 = one day leeway
# 2 = don't test
def temporal_constraint_match(event_entry, constraint_entry, leeway = 0):
    if leeway > 1:
        # no testing, assume it fits.
        return True

    if constraint_entry is None:
        # no temporal constraint, no problem
        return True

    # does it have a temporal entry?
    if "ldcTime" not in event_entry:
        # we were looking for a temporal entry but did not find one.
        # this is a constraint violation.
        return False

    # we have a match if at least one of the temporal statements of the entry for qvar_filler match
    for ldc_time in event_entry["ldcTime"]:
        if temporal_constraint_match_one_event_date(ldc_time, constraint_entry, leeway):
            return True
        
    # we found no time for our event that matched the temporal constraint
    return False


def temporal_constraint_match_one_event_date(event_date, constraint_entry, leeway):
    if "start" in event_date:
        for starttime in event_date["start"]:
            if "end" in event_date:
                for endtime in event_date["end"]:
                    if temporal_constraint_match_one(starttime, endtime, constraint_entry, leeway):
                        # this time matched the temporal constraint
                        return True
            else:
                # we don't have an event end date. 
                if temporal_constraint_match_one(starttime, None, constraint_entry, leeway):
                    return True
    else:
        # we don't have an event start date
        if "end" in event_date:
            for endtime in event_date["end"]:
                if temporal_constraint_match_one(None, endtime, constraint_entry, leeway):
                    # this time matched the temporal constraint
                    return True

    # odd event entry: no start date or end date. consider this unmatched.
    return False


def temporal_constraint_match_one(eventstart_json, eventend_json, constraint_date, leeway = 0):
    eventstart_json = json_unlist(eventstart_json)
    eventend_json = json_unlist(eventend_json)

    if eventstart_json is not None:
        eventstart = AidaIncompleteDate(eventstart_json.get("year", None), eventstart_json.get("month", None), eventstart_json.get("day", None))
    else:
        eventstart = None

    if eventend_json is not None:
        eventend = AidaIncompleteDate(eventend_json.get("year", None), eventend_json.get("month", None), eventend_json.get("day", None))
    else:
        eventend = None

    # event has "start" and "end", which can be given a timeType of ON/BEFORE/AFTER a date.
    # the query has "startTime" and "endTime", no time type
    add_a_day = (leeway > 0)

    if "start_time" in constraint_date and eventstart is not None:
        # we can compare start times. otherwise, assume compatibility.

        if eventstart_json["timeType"] == "BEFORE" and  not(constraint_date["start_time"].is_before(eventstart,add_a_day)):
            # event start before t and qt.start_time not before t: constraint violation
            return False

        if eventstart_json["timeType"] == "AFTER" and not(eventstart.is_before(constraint_date["start_time"], add_a_day)): 
            # event start after t and qt.start_time before t: constraint violation
            return False
        
        if eventstart_json["timeType"] == "ON" and not(eventstart.is_eq(constraint_date["start_time"], add_a_day)):
            # event start and qt.start_time not equal
            return False

    if "end_time" in constraint_date and eventend is not None:
        # we can compare end times. otherwise, assume compatibility

        if eventend_json["timeType"]  == "BEFORE" and not(constraint_date["end_time"].is_before(eventend, add_a_day)):
            # event end before t and t before qt.end_time: constraint violation
            return False
        if eventend_json["timeType"] == "AFTER" and not(eventend.is_before(constraint_date["end_time"], add_a_day)):
            # event end after t and qt.end_time before t: constraint violation
            return False
        if eventend_json["timeType"] == "ON" and not(eventend.is_eq(constraint_date["end_time"], add_a_day)):
            # event end and qt.end_time not  equal
            return False    

    return True

def json_unlist(ldcdate):
    retv = { }
    for key, val in ldcdate.items():
        retv[key] = val[0]
    return retv
        
