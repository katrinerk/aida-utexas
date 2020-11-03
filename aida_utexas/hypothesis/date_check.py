"""
Author: Katrin Erk, May 2019
- Check temporal constraints

Update: Pengxiang Cheng, Aug 2020
- Refactoring and clean-up
- Needs further rewriting to accommodate to new M36 temporal specifications
"""

import calendar
import datetime
import logging
from typing import Dict, Optional

from aida_utexas.aif.json_graph import ERENode


# AIDA temporal info handling,
# with missing entries for days/months/years
class AidaIncompleteDate:
    def __init__(self, year: Optional[int] = None, month: Optional[int] = None,
                 day: Optional[int] = None):
        self.year = year
        self.month = month
        self.day = day

    def to_json(self):
        return {
            'year': self.year,
            'month': self.month,
            'day': self.day
        }

    @classmethod
    def from_json(cls, date_json: Dict):
        return cls(
            year=date_json.get('year', None),
            month=date_json.get('month', None),
            day=date_json.get('day', None)
        )

    @staticmethod
    def _check_date_str(date, date_field):
        if date is not None and not isinstance(date, int):
            if isinstance(date, str) and date.isdigit():
                date = int(date)
            else:
                if not (isinstance(date, str) and (date.strip("X") == "")):
                    logging.warning(f'unexpected type for {date_field}: {date}, {type(date)}')
                date = None
        return date

    # helper method to indicate whether the time construct actually represents a temporal query
    def is_query(self):
        if self.year is not None and self.year < 0:
            return True
        if self.month is not None and self.month < 0:
            return True
        if self.day is not None and self.day < 0:
            return True
        return False

    # interpret the AidaIncompleteDate to a python datetime.date object, with optional relaxation
    # if time_type is 'AFTER', convert an underspecified date to its earliest possible date
    # if time_type is 'BEFORE', convert an underspecified date to its latest possible date
    # in all other cases, if any of the year, month, or day is unspecified, return None
    def interpret(self, time_type: str = None):
        # we shouldn't really be calling the interpret() function on temporal queries
        if self.is_query():
            return None

        year, month, day = self.year, self.month, self.day

        # KATRIN QUICK FIX, don't know why we sometimes get year = 9999, month =12, day= 31
        if year >= 9900:
            year = 9000
            logging.warning(f'KATRIN2 {year} {month} {day}')
        elif year <= 100:
            year = 101
            logging.warning(f'KATRIN2 {year} {month} {day}')

        # if year is not specified, this is an unspecified date
        if year is None:
            return None
        # if month is not specified
        if month is None:
            # if day is specified (should not happen), this is an unspecified date
            if day is not None:
                return None
            # otherwise, if time_type is AFTER, set the month to the first month of the year
            if time_type == 'AFTER':
                month = 1
            # if time_type is BEFORE, set the month to the last month of the year
            elif time_type == 'BEFORE':
                month = 12
        # if day is not specified
        if day is None:
            # if time_type is AFTER, set the day to the first day of the month
            if time_type == 'AFTER':
                day = 1
            # if time_type is BEFORE, set the day to the last day of the month
            elif time_type == 'BEFORE':
                day = calendar.monthrange(year, month)[1]

        if month is None or day is None:
            return None
        else:
            return datetime.date(year, month, day)

    # add a day to the date
    def add_day(self):
        # convert to python date, without specifying time_type (no relaxation)
        curr_date = self.interpret()
        # if it is an incomplete date, return the same date
        if curr_date is None:
            return AidaIncompleteDate(self.year, self.month, self.day)

        # otherwise, add a day to the current date
        next_date = curr_date + datetime.timedelta(1)
        return AidaIncompleteDate(next_date.year, next_date.month, next_date.day)

    # can t1 be before t2?
    def is_before(self, date2, add_a_day: bool = False):
        # use AFTER as the time_type, as we assume the earliest possible date for incomplete date1
        p_date1 = self.interpret(time_type='AFTER')
        # use BEFORE as the time_type, as we assume the latest possible date for incomplete date2
        p_date2 = date2.interpret(time_type='BEFORE')

        # if either date1 or date2 is still underspecified with relaxation, return True
        if p_date1 is None or p_date2 is None:
            return True

        # if add_a_day is True, add a day to p_date2
        if add_a_day:
            p_date2 = p_date2 + datetime.timedelta(1)

        # otherwise, check if p_date1 is before or same as p_date2
        return p_date1 <= p_date2


# returns true if any one of the ldcTimes of the ERE node matches the temporal constraint,
# or if there is no temporal constraint
# leeway levels: 0 = no leeway, 1 = one day leeway, 2 = don't test
def temporal_constraint_match(ere_node: ERENode, constraint_entry: Dict[str, AidaIncompleteDate],
                              leeway: int = 0):
    # no testing, assume it fits.
    if leeway > 1:
        return True

    # no temporal constraint, no problem
    if constraint_entry is None:
        return True

    # we were looking for ldcTime attributes of the ERE node, but did not find one: violation.
    if not ere_node.ldcTime:
        return False

    # get the start time and the end time from the temporal constraint
    constraint_start = constraint_entry.get('start_time', None)
    constraint_end = constraint_entry.get('end_time', None)

    # we have a match if at least one ldcTime of the ERE node matches the temporal constraint
    for ldc_time in ere_node.ldcTime:
        # extract the [T1, T2, T3, T4] tuple from ldcTime
        start_after, start_before, end_after, end_before = None, None, None, None
        for start_time in ldc_time.get('start', [None]):
            if start_time is not None:
                if start_time['timeType'] == 'AFTER':
                    start_after = AidaIncompleteDate.from_json(start_time)
                if start_time['timeType'] == 'BEFORE':
                    start_before = AidaIncompleteDate.from_json(start_time)
        for end_time in ldc_time.get('end', [None]):
            if end_time is not None:
                if end_time['timeType'] == 'AFTER':
                    end_after = AidaIncompleteDate.from_json(end_time)
                if end_time['timeType'] == 'BEFORE':
                    end_before = AidaIncompleteDate.from_json(end_time)

        # check if the constraint start time is between T1 and T2, and if the constraint end time
        # is between T3 and T4, allowing a possible leeway (add a day)
        if time_range_match(start_after, start_before, constraint_start, leeway) \
                and time_range_match(end_after, end_before, constraint_end, leeway):
            return True

    # we found no ldcTime of the ERE node that matches the temporal constraint
    return False


# check if the constraint_time is in between of ldc_time_after and ldc_time_before,
# allowing a possible leeway
def time_range_match(ldc_time_after: AidaIncompleteDate, ldc_time_before: AidaIncompleteDate,
                     constraint_time: AidaIncompleteDate, leeway: int = 0):
    if constraint_time is None:
        return True

    # if the temporal info is a query, and there is no ldcTime associated with the ERE, treat it
    # as a no match, which will lead to a penalty
    if constraint_time.is_query():
        if ldc_time_after is None and ldc_time_before is None:
            return False

    # whether to allow add_a_day leeway
    add_a_day = leeway > 0

    # if constraint_time is None, there is nothing to check
    if constraint_time is not None:
        # if ldc_time_after is not None
        if ldc_time_after is not None:
            # make sure that ldc_time_after is before constraint_time
            if not ldc_time_after.is_before(constraint_time, add_a_day=add_a_day):
                return False

        # if ldc_time_before is not None
        if ldc_time_before is not None:
            # make sure that constraint_time is before ldc_time_before
            if not constraint_time.is_before(ldc_time_before, add_a_day=add_a_day):
                return False

    return True
