import { Component } from '@angular/core';
import { ProbObjectService } from '../../services/prob-object.service';
import { ProbObject } from '../../models/predict-genre-response.model';

@Component({
  selector: 'app-result-view',
  imports: [],
  templateUrl: './result-view.component.html',
  styleUrl: './result-view.component.scss',
})
export class ResultViewComponent {
  probObject: ProbObject | null = null;

  constructor(private probObjectService: ProbObjectService) {}
  ngOnInit() {
    this.probObject = this.probObjectService.getProbObject();
    console.log('Results: ', this.probObject);
    if (!this.probObject) {
      console.warn('No classification result found.');
    }
  }
}
