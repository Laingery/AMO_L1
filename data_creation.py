import math
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class WinterTemperatureGenerator():

    def __init__(self, numberGenDey, safeDataSet = True, showGraph = False):
        self.numberGenDey = numberGenDey
        self.current_day = 1

        self.showGraph = showGraph
        self.safeDataSet = safeDataSet

        # self.generate_temperature_profile()
        self.current_dataframe = pd.DataFrame()  # Храним текущий DataFrame

        self.model = ThermalModel()



    def startGen(self):
        for i in range(1, numberGenDey + 1):
            self.current_day = i
            self.generate_temperature_profile()

        if self.safeDataSet:
            self.save_data_pandas()

    def generate_temperature_profile(self):
        """Генерация суточного профиля температуры с шагом 5 минут"""
        night_min = -20
        day_max = -6

        points_per_day = 288
        self.original_temperatures = []

        for step in range(points_per_day):
            hour = step / 12

            if 3 <= hour <= 5:
                temp = night_min + random.uniform(-1, 1)
            elif 13 <= hour <= 15:
                temp = day_max + random.uniform(-1, 1)
            else:
                hour_normalized = (hour - 4) / 24.0
                temp = night_min + (day_max - night_min) * (0.5 + 0.5 * np.sin(2 * np.pi * hour_normalized - np.pi / 2))
                temp += random.uniform(-0.3, 0.3)

            self.original_temperatures.append(round(temp, 1))

        self.original_temperatures = self.smooth_temperature(self.original_temperatures)

        self.original_temperatures = self.apply_anomalies(self.original_temperatures)

        self.temperatures = self.original_temperatures.copy()

        # Создаем DataFrame
        self.create_dataframe()

    def smooth_temperature(self, temps):
        """Сглаживание температурного профиля"""
        smoothed = temps.copy()
        for i in range(1, len(temps) - 1):
            smoothed[i] = (temps[i - 1] + temps[i] + temps[i + 1]) / 3
        return smoothed

    def create_dataframe(self):
        """Создание DataFrame из текущих данных"""
        data = {
            'Номер_дня': [],
            'Время_дня': [],
            'Температура_°C': [],
            'Температура_A1_°C': [],
            'Температура_A2_°C': [],
            'Температура_A3_°C': []
        }

        for step, temp in enumerate(self.temperatures):
            hours, minutes = self.get_time_from_step(step)
            time_str = f"{hours:02d}:{minutes:02d}"
            data['Номер_дня'].append(self.current_day)
            data['Время_дня'].append(time_str)
            data['Температура_°C'].append(temp)
            # Обновление модели
            T_A1, T_A2, T_A3 = self.model.update(temp, step)
            data['Температура_A1_°C'].append(T_A1)
            data['Температура_A2_°C'].append(T_A2)
            data['Температура_A3_°C'].append(T_A3)

        dataSets = ["нет", "Температура_°C", "Температура_A1_°C", "Температура_A2_°C", "Температура_A3_°C"]
        weights = [60, 10, 10, 10, 10]
        noise_dataSet = random.choices(dataSets, weights=weights, k=1)[0]

        # print(noise_dataSet)

        if noise_dataSet == "Температура_°C":
            data['Температура_°C'] = self.apply_noise(data['Температура_°C'])
        elif noise_dataSet == "Температура_A1_°C":
            data['Температура_A1_°C'] = self.apply_noise(data['Температура_A1_°C'])
        elif noise_dataSet == "Температура_A2_°C":
            data['Температура_A2_°C'] = self.apply_noise(data['Температура_A2_°C'])
        elif noise_dataSet == "Температура_A3_°C":
            data['Температура_A3_°C'] = self.apply_noise(data['Температура_A3_°C'])

        if self.current_dataframe.empty:
            self.current_dataframe = pd.DataFrame(data)
        else:
            self.current_dataframe = pd.concat([self.current_dataframe, pd.DataFrame(data)], ignore_index=True)

        if self.showGraph:
            plt.figure(figsize=(14, 6))
            plt.plot(self.current_dataframe['Температура_°C'], 'r--', linewidth=2, label='1')
            plt.plot(self.current_dataframe['Температура_A1_°C'], 'b-', linewidth=2, label='Температура внешнего куба')
            plt.plot(self.current_dataframe['Температура_A2_°C'], 'g-', linewidth=2, label='Температура внутри куба')
            plt.plot(self.current_dataframe['Температура_A3_°C'], 'm-', linewidth=2, label='Алюминий c наг.')

            plt.xlabel('Время (отрезки по 5 минут)', fontsize=12)
            plt.ylabel('Температура (°C)', fontsize=12)
            plt.title('Изменение температуры внутри системы', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.show()

    def get_time_from_step(self, step):
        """Преобразование шага в часы и минуты"""
        total_minutes = step * 5
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return hours, minutes

    def save_data_pandas(self):
        """Сохранение данных с использованием pandas"""
        if self.current_dataframe is None or self.current_dataframe.empty:
            return

        if not os.path.exists("test"):
            os.makedirs("test")

        if not os.path.exists("train"):
            os.makedirs("train")

        filename = f"temperature_data_day{self.current_day}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if filename:
            # Создаем копию DataFrame для сохранения с дополнительной информацией
            df_to_save = self.current_dataframe.copy()
            # Сохраняем в CSV с разделителем точка с запятой для корректного отображения в Excel
            df_to_save.to_csv(f"{filename}.csv", index=False, encoding='utf-8-sig', sep=';')
        else:
            return

        train, test = train_test_split(self.current_dataframe, test_size=0.2, random_state=42)
        # test.to_csv(f"./test/{filename}_test.csv", index=False, encoding='utf-8-sig', sep=';')
        # train.to_csv(f"./train/{filename}_train.csv", index=False, encoding='utf-8-sig', sep=';')
        test.to_csv(f"./test/test.csv", index=False, encoding='utf-8-sig', sep=';')
        train.to_csv(f"./train/train.csv", index=False, encoding='utf-8-sig', sep=';')

        return filename



    def apply_noise(self,temps):
        noiseSet = ["нет", "Гауссовский", "Равномерный", "Импульсный", "Периодический"]
        weights = [60, 10, 10, 10, 10]
        noise_type = random.choices(noiseSet, weights=weights, k=1)[0]

        if noise_type == "нет":
            return temps

        noiseDate = temps.copy()

        intensity = random.uniform(0.5, 2.5)
        noise = np.zeros(len(noiseDate))

        # print(noise_type)

        if noise_type == "Гауссовский":
            noise = NoiseGenerator.gaussian_noise(len(noiseDate), 0, intensity)
        elif noise_type == "Равномерный":
            noise = NoiseGenerator.uniform_noise(len(noiseDate), -intensity, intensity)
        elif noise_type == "Импульсный":
            prob = 2
            amp = intensity * 3
            noise = NoiseGenerator.impulse_noise(len(noiseDate), prob, amp)
        elif noise_type == "Периодический":
            noise = NoiseGenerator.periodic_noise(len(noiseDate), intensity, 2)

        for i in range(len(noiseDate)):
            noiseDate[i] = round(noiseDate[i] + noise[i], 1)

        return noiseDate

    def apply_anomalies(self, temps):
        """Применение аномалий к данным"""

        anomalySet = ["нет", "Холодное вторжение", "Тепловая аномалия",
                                    "Резкое похолодание", "Усиление ночного мороза"]
        weights = [50, 10, 10, 10, 20]
        anomaly_type = random.choices(anomalySet, weights=weights, k=1)[0]

        # print(anomaly_type)

        if anomaly_type == "нет":
            return temps

        anomalies = temps.copy()

        intensity = random.uniform(-4, -10)
        duration = random.uniform(2, 6)

        if anomaly_type == "Холодное вторжение":
            anomalies = AnomalyGenerator.cold_spell(anomalies, intensity, duration)
        elif anomaly_type == "Тепловая аномалия":
            anomalies = AnomalyGenerator.heat_anomaly(anomalies, abs(intensity), duration)
        elif anomaly_type == "Резкое похолодание":
            anomalies = AnomalyGenerator.sharp_drop(anomalies, intensity, duration)
        elif anomaly_type == "Усиление ночного мороза":
            anomalies = AnomalyGenerator.night_frost_anomaly(anomalies, intensity)

        anomalies = [round(temp, 1) for temp in anomalies]

        return anomalies

class ThermalModel:
    def __init__(self):

        # Начальные температуры
        self.T_A1 = -6.0  # температура обьёма A1
        # Начальные температуры
        self.T_A2 = -6.0  # температура обьёма A2
        # Начальные температуры
        self.T_A3 = -6.0  # температура обьёма A3

        # ТЕПЛОФИЗИЧЕСКИЕ СВОЙСТВА
        rho_air = 1.225  # кг/м³
        c_air = 1005  # Дж/кг·К
        self.alpha_air = 10

        # ТЕПЛОФИЗИЧЕСКИЕ СВОЙСТВА (Алюминий)
        al_rho_wall = 2700  # плотность (кг/м³)
        al_c_wall = 900  # теплоёмкость (Дж/кг·К)
        al_lambda_wall = 220.0  # теплопроводность (Вт/м·К)

        # ТЕПЛОФИЗИЧЕСКИЕ СВОЙСТВА (Стеклокомпозит)
        comp_rho_wall = 1800  # плотность (кг/м³)
        comp_c_wall = 800  # теплоёмкость (Дж/кг·К)
        comp_lambda_wall = 0.4  # теплопроводность (Вт/м·К)

        # ========== Короб A3 ==========
        self. power_watts = 200.0  # теплавая мощность нагрузки
        # размер
        A3_side_outer = 0.4  # внешняя сторона куба (м)
        A3_wall_thick = 0.02  # толщина стенки 2 см = 0.02 м
        self.A3_s_area = 6 * (A3_side_outer ** 2)  # Площадь поверхности

        A3_v_outer = A3_side_outer ** 3
        A3_v_inner = (A3_side_outer - 2 * A3_wall_thick) ** 3
        A3_m_al = (A3_v_outer - A3_v_inner) * al_rho_wall  # масса алюминия

        # ТЕПЛОЁМКОСТИ
        A3_C_wall = A3_m_al * al_c_wall  # теплоёмкость стенок
        A3_C_air = rho_air * A3_v_inner * c_air  # теплоёмкость воздуха
        A3_C_total = A3_C_wall + A3_C_air  # общая теплоёмкость

        self.A3_delta_t_steady = self.power_watts / (self.alpha_air * self.A3_s_area)

        # Коэффициент скорости изменения температуры
        self.A3_k = (self.alpha_air * self.A3_s_area) / A3_C_total

        # ========== Короб A2 ==========

        # размер
        A2_side_outer = 0.7  # внешняя сторона куба (м)
        A2_wall_thick = 0.03  # толщина стенки 0,5 см = 0.005 м
        A2_s_area = 6 * (A2_side_outer ** 2)  # Площадь поверхности
        A2_side_inner = A2_side_outer - 2 * A2_wall_thick  # внутренняя сторона (м)

        # Объёмы
        A2_volume_inner = A2_side_inner ** 3  # объём воздуха внутри (м³)
        A2_volume_wall = A2_side_outer ** 3 - A2_volume_inner  # объём материала стенок (м³)

        # ========== ТЕПЛОЁМКОСТИ ==========
        A2_C_wall = comp_rho_wall * A2_volume_wall * comp_c_wall  # теплоёмкость стенок
        A2_C_air = rho_air * A2_volume_inner * c_air  # теплоёмкость воздуха
        self.A2_C_total = A2_C_wall + A2_C_air  # общая теплоёмкость

        A2_A_inner = 6 * A2_side_inner ** 2  # м²

        A2_R_wall = A2_wall_thick / comp_lambda_wall

        self.A2_U = A2_A_inner / A2_R_wall

        # ========== Короб А1 ==========

        # размер
        A1_side_outer = 5.0  # внешняя сторона куба (м)
        A1_wall_thick = 0.05  # толщина стенки
        A1_s_area = 6 * (A1_side_outer ** 2)  # Площадь поверхности
        A1_side_inner = A1_side_outer - 2 * A1_wall_thick  # внутренняя сторона (м)

        # Объёмы
        A1_volume_inner = A1_side_inner ** 3  # объём воздуха внутри (м³)
        A1_volume_wall = A1_side_outer ** 3 - A1_volume_inner  # объём материала стенок (м³)

        # ========== ТЕПЛОЁМКОСТИ ==========
        A1_C_wall = comp_rho_wall * A1_volume_wall * comp_c_wall  # теплоёмкость стенок
        A1_C_air = rho_air * A1_volume_inner * c_air  # теплоёмкость воздуха
        self.A1_C_total = A1_C_wall + A1_C_air  # общая теплоёмкость

        A1_A_inner = 6 * A1_side_inner ** 2  # м²

        A1_R_wall = A1_wall_thick / comp_lambda_wall

        self.A1_U = A1_A_inner / A1_R_wall


    def update(self, T_outside, time):
        """
                Обновление температур
                T_outside: уличная температура (°C)
        """

        A1_Q = self.A1_U * (T_outside - self.T_A1)
        A1_dT = (A1_Q * 300) / self.A1_C_total
        self.T_A1 = self.T_A1 + A1_dT

        A2_Q = self.A2_U * (self.T_A1 - self.T_A2)
        A2_dT = (A2_Q * 300) / self.A2_C_total
        self.T_A2 = self.T_A2 + A2_dT

        self.A3_delta_t_steady = (80 if (time >= 240 or time < 72) else 20) /(self.alpha_air * self.A3_s_area)

        t_steady = self.T_A2 + self.A3_delta_t_steady
        self.T_A3 = t_steady + (self.T_A2 - t_steady) * math.exp(-self.A3_k * 300)

        A2_Q = self.A2_U * (self.T_A3 - self.T_A2)
        A2_dT = (A2_Q * 300) / self.A2_C_total
        self.T_A2 = self.T_A2 + A2_dT

        return self.T_A1, self.T_A2, self.T_A3

class NoiseGenerator:
    """Класс для генерации различных типов шумов"""

    @staticmethod
    def gaussian_noise(size, mean=0, std=1):
        """Гауссовский шум"""
        return np.random.normal(mean, std, size)

    @staticmethod
    def uniform_noise(size, low=-1, high=1):
        """Равномерный шум"""
        return np.random.uniform(low, high, size)

    @staticmethod
    def impulse_noise(size, probability=0.05, amplitude=5):
        """Импульсный шум (выбросы)"""
        noise = np.zeros(size)
        for i in range(size):
            if random.random() < probability:
                noise[i] = random.choice([-amplitude, amplitude]) * random.uniform(0.5, 1)
        return noise

    @staticmethod
    def periodic_noise(size, amplitude=1, frequency=2):
        """Периодический шум"""
        t = np.linspace(0, 2 * np.pi * frequency, size)
        return amplitude * np.sin(t)

class AnomalyGenerator:
    """Класс для генерации различных аномалий"""

    @staticmethod
    def cold_spell(temperatures, intensity=-8, duration_hours=4):
        """Холодное вторжение"""
        temps = temperatures.copy()
        points_count = len(temperatures)
        duration_points = int(duration_hours * 12)
        center = random.randint(int(points_count * 0.25), int(points_count * 0.75))
        start = max(0, center - duration_points // 2)
        end = min(points_count, center + duration_points // 2)
        for i in range(start, end):
            progress = (i - start) / (end - start) if end > start else 1
            factor = 1 - 2 * abs(progress - 0.5)
            temps[i] += intensity * max(0, factor)
        return temps

    @staticmethod
    def heat_anomaly(temperatures, intensity=5, duration_hours=3):
        """Тепловая аномалия"""
        temps = temperatures.copy()
        points_count = len(temperatures)
        duration_points = int(duration_hours * 12)
        center = random.randint(int(points_count * 0.4), int(points_count * 0.8))
        start = max(0, center - duration_points // 2)
        end = min(points_count, center + duration_points // 2)
        for i in range(start, end):
            progress = (i - start) / (end - start) if end > start else 1
            factor = 1 - 2 * abs(progress - 0.5)
            temps[i] += intensity * max(0, factor)
        return temps

    @staticmethod
    def sharp_drop(temperatures, drop=-12, recovery_hours=6):
        """Резкое похолодание с постепенным восстановлением"""
        temps = temperatures.copy()
        points_count = len(temperatures)
        drop_point = random.randint(int(points_count * 0.3), int(points_count * 0.7))
        recovery_points = int(recovery_hours * 12)
        for i in range(drop_point, min(points_count, drop_point + recovery_points)):
            progress = (i - drop_point) / recovery_points if recovery_points > 0 else 1
            temps[i] += drop * (1 - progress)
        return temps

    @staticmethod
    def night_frost_anomaly(temperatures, extra_cold=-5):
        """Усиление ночного мороза"""
        temps = temperatures.copy()
        points_count = len(temperatures)
        night_points = int(8 * 12)
        for i in range(night_points):
            hour = i / 12
            factor = 1 - abs(hour - 4) / 4 if hour != 4 else 1
            temps[i] += extra_cold * max(0, factor)
        return temps

numberGenDey: int = 90

Generator = WinterTemperatureGenerator(numberGenDey)
Generator.startGen()



