import { Component } from '@angular/core';
import { ProbObjectService } from '../../services/prob-object.service';
import { ProbObject } from '../../models/predict-genre-response.model';
import { CommonModule } from '@angular/common';
import { PieChartComponent } from "../../components/pie-chart/pie-chart.component";
import { Router } from '@angular/router';

@Component({
  selector: 'app-result-view',
  standalone: true,
  imports: [CommonModule, PieChartComponent],
  templateUrl: './result-view.component.html',
  styleUrls: ['./result-view.component.scss'],
})
export class ResultViewComponent {
  propObject: ProbObject | null = null;
  sortedPropEntries: [string, number][] = [];
  pieChartData: { labels: string[]; values: number[] } | null = null;
  songName: string | null = null;

  constructor(
    private probObjectService: ProbObjectService,
    private router: Router) {}

  ngOnInit() {
    this.propObject = this.probObjectService.getProbObject();
    this.songName = this.probObjectService.getFileName();
    console.log('Results: ', this.propObject);
    console.log('Song Name: ', this.songName);

    if (this.propObject) {
      this.pieChartData = {
        labels: Object.keys(this.propObject),
        values: Object.values(this.propObject),
      }
    } else {
      console.warn('No classification result found.');
    }
  }

  goBack() {
    this.router.navigate(['/predict-genre']);
  }
}
